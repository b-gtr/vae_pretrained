import gym
from gym import spaces
import numpy as np
import math
import random
import cv2
import carla
import threading

# ---------------------------------
# Hilfsfunktionen
# ---------------------------------
def vector_2d(vec_carla):
    """Konvertiere carla.Vector3D -> (x, y) in float."""
    return np.array([vec_carla.x, vec_carla.y], dtype=np.float32)

def distance_2d(a, b):
    """Euklidischer 2D-Abstand."""
    return float(np.linalg.norm(a - b))

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def compute_lateral_offset(vehicle_transform, waypoint_transform):
    """
    Berechnet die laterale Distanz (seitlicher Versatz) des Fahrzeugs zum 'center' 
    des Waypoints (fahrbare Spur). Ignoriert z-Höhenunterschiede.
    """
    veh_loc = vehicle_transform.location
    wp_loc = waypoint_transform.location

    # 1) Vektor vom Waypoint-Zentrum zum Fahrzeug
    dx = veh_loc.x - wp_loc.x
    dy = veh_loc.y - wp_loc.y

    # 2) Vorwärtsrichtung des Waypoints
    forward = waypoint_transform.get_forward_vector()
    fx, fy = forward.x, forward.y

    # 3) "Quer"-Abstand über 2D-Kreuzprodukt 
    cross_val = dx * fy - dy * fx
    return cross_val

def is_waypoint_behind(vehicle_transform, waypoint_transform):
    """
    Prüft, ob der Waypoint 'hinter' dem Fahrzeug liegt.
    Das geht z.B. über das Skalarprodukt zwischen Fahrzeug-Vorwärtsrichtung
    und dem Vektor Fahrzeug->Waypoint. Ist < 0 => Waypoint liegt hinter uns.
    """
    veh_loc = vehicle_transform.location
    wp_loc = waypoint_transform.location

    forward = vehicle_transform.get_forward_vector()
    to_waypoint = wp_loc - veh_loc

    dot = forward.x * to_waypoint.x + forward.y * to_waypoint.y + forward.z * to_waypoint.z
    return dot < 0.0

# ---------------------------------
# Sensoren
# ---------------------------------
class CollisionSensor:
    def __init__(self, vehicle, blueprint_library, world):
        self.vehicle = vehicle
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.history = []
        self._listen()

    def _listen(self):
        self.sensor.listen(lambda event: self.history.append(event))

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()

    def get_history(self):
        return self.history


class CameraSensor:
    def __init__(self, vehicle, blueprint_library, world, callback):
        self.vehicle = vehicle
        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')
        transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.sensor = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
        self.callback = callback
        self._listen()

    def _listen(self):
        self.sensor.listen(self.callback)

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()


# ---------------------------------
# Haupt-Env
# ---------------------------------
class CarlaGymEnv(gym.Env):
    """
    Environment, das:
      - semantische Kamera in (480,640,1) liefert,
      - Distanz zur Fahrbahnmitte,
      - GPS-Koordinate zum nächsten Waypoint,
      - eigene GPS-Koordinate,
      - Geschwindigkeit
    als Dictionary-Observation zurückgibt.
    
    Zusätzlich haben wir ein Step-Limit (max_steps) eingebaut, 
    um die Episodenlänge zu begrenzen.
    """
    def __init__(self, host='localhost', port=2000, display=True, max_steps=1000):
        super().__init__()
        self.host = host
        self.port = port
        self.display = display  # Falls True, zeige Kamera-Bild per cv2 an.

        # Anzahl der Schritte bis Episode endet (falls keine andere Termination eintritt)
        self.max_steps = max_steps
        self.current_step = 0

        # Verbinde mit CARLA
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)

        # Lade die Welt
        self.world = self.client.load_world('Town01')
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()

        # Synchronous Mode einstellen
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

        # Thread-Lock für Kamera
        self.image_lock = threading.Lock()
        self.camera_image = None

        # Actor placeholders
        self.vehicle = None
        self.collision_sensor = None
        self.camera_sensor = None

        # Wir halten den "nächsten Waypoint" fest (carla.Waypoint)
        self.next_waypoint = None

        # ACTION SPACE: [steer, throttle] in [-0.5, +0.5]
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(2,), dtype=np.float32)

        # OBSERVATION SPACE: Dictionary
        self.observation_space = spaces.Dict({
            "segmentation": spaces.Box(
                low=0.0, high=1.0, shape=(480, 640, 1), dtype=np.float32
            ),
            "dist_center": spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            "gps_next_waypoint": spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            ),
            "gps_own": spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            ),
            "speed": spaces.Box(
                low=0.0, high=np.inf, shape=(1,), dtype=np.float32
            ),
        })

        # Erste Episode initialisieren
        self.reset()

    # ---------------------------------
    # Hilfsfunktionen
    # ---------------------------------
    def _init_vehicle_sensors(self):
        # Fahrzeug
        vehicle_bp = self.blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
        spawn_point = random.choice(self.spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Kollision
        self.collision_sensor = CollisionSensor(self.vehicle, self.blueprint_library, self.world)

        # Kamera
        def camera_callback(image):
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))  # => (480,640,4)

            # Red-Kanal hat die semantischen Labels
            labels = array[:, :, 2].astype(np.float32)
            labels /= 22.0  # Normierung auf [0,1]

            labels = np.expand_dims(labels, axis=-1)  # (480,640) => (480,640,1)
            with self.image_lock:
                self.camera_image = labels

        self.camera_sensor = CameraSensor(
            self.vehicle, self.blueprint_library, self.world, camera_callback
        )

        # Warmlaufen (einige Ticks)
        for _ in range(10):
            self.world.tick()

        # Waypoint setzen
        self._pick_next_waypoint()

    def _clear_actors(self):
        if self.camera_sensor is not None:
            self.camera_sensor.destroy()
            self.camera_sensor = None
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
            self.collision_sensor = None
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None
        self.camera_image = None

    def _pick_next_waypoint(self):
        """Setzt self.next_waypoint auf eine zufällige Option in Fahrtrichtung."""
        if not self.vehicle:
            return

        veh_transform = self.vehicle.get_transform()
        current_wp = self.map.get_waypoint(
            veh_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        possible_next = current_wp.next(5.0)
        if len(possible_next) == 0:
            # Kein nächster WP gefunden => NOP
            return
        # Falls Verzweigungen, wähle Zufällig eine
        self.next_waypoint = random.choice(possible_next)

    def _remove_all_vehicles(self):
        """Löscht alle Vehicles in der aktuellen Welt."""
        actors = self.world.get_actors().filter("*vehicle*")
        for a in actors:
            a.destroy()

    def get_vehicle_speed(self):
        """Geschwindigkeit in m/s."""
        if not self.vehicle:
            return 0.0
        vel = self.vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    # ---------------------------------
    # Gym-Methoden
    # ---------------------------------
    def reset(self):
        # Schrittzähler zurücksetzen
        self.current_step = 0

        # Alte Actoren löschen
        self._clear_actors()

        # Neu spawnen
        self._init_vehicle_sensors()

        return self._get_obs()

    def step(self, action):
        # Schrittzähler inkrementieren
        self.current_step += 1

        # Action = [steer, throttle], klippen in [-0.5, 0.5]
        steer = float(clamp(action[0], -0.5, 0.5))
        throttle = float(clamp(action[1], -0.5, 0.5))

        # Throttle in [0,0.5] umwandeln
        throttle = (throttle + 0.5)  # => [0..1], falls -0.5..+0.5
        throttle = clamp(throttle, 0.0, 0.5)

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        self.vehicle.apply_control(control)

        # Einen Tick simulieren
        self.world.tick()

        reward, done, info = self._compute_reward_done_info()

        # Zeitablauf: Falls max. Schrittzahl erreicht => done
        if self.current_step >= self.max_steps:
            done = True
            info["time_out"] = True

        # Waypoint hinter uns? => Episode beenden und alles löschen
        if is_waypoint_behind(self.vehicle.get_transform(), self.next_waypoint.transform):
            done = True
            info["waypoint_behind"] = True
            self._remove_all_vehicles()

        obs = self._get_obs()
        return obs, reward, done, info

    def _compute_reward_done_info(self):
        info = {}
        done = False
        reward = 0.0

        # 1) Kollision
        if len(self.collision_sensor.get_history()) > 0:
            reward = -1.0
            done = True
            info["collision"] = True
            return reward, done, info

        # 2) Distanz zur Mitte
        lateral_offset = compute_lateral_offset(
            self.vehicle.get_transform(),
            self.map.get_waypoint(self.vehicle.get_transform().location).transform
        )
        offset_magnitude = abs(lateral_offset)
        max_offset = 2.0
        if offset_magnitude >= max_offset:
            dist_center_reward = -0.5
        else:
            dist_center_reward = 0.5 * (1.0 - offset_magnitude / max_offset)

        # 3) Geschwindigkeit
        speed = self.get_vehicle_speed()
        if speed < 0.1:
            speed_reward = -0.3
        else:
            capped_speed = min(speed, 10.0)  # 10 m/s = ~36 km/h
            speed_reward = 0.5 * (capped_speed / 10.0)

        # Summe
        reward = dist_center_reward + speed_reward
        # Clampen auf [-1, +1]
        reward = clamp(reward, -1.0, 1.0)

        return reward, done, info

    def _get_obs(self):
        with self.image_lock:
            if self.camera_image is None:
                seg_img = np.zeros((480, 640, 1), dtype=np.float32)
            else:
                seg_img = self.camera_image.copy()

        # Falls display=True => zeige das Bild
        if self.display:
            self._show_image(seg_img)

        # Distanz zur Fahrbahnmitte
        lateral_offset = compute_lateral_offset(
            self.vehicle.get_transform(),
            self.map.get_waypoint(self.vehicle.get_transform().location).transform
        )
        dist_center = np.array([lateral_offset], dtype=np.float32)

        # GPS (x,y) des Fahrzeugs
        veh_loc = self.vehicle.get_transform().location
        gps_own = np.array([veh_loc.x, veh_loc.y], dtype=np.float32)

        # GPS (x,y) des nächsten Waypoints
        if self.next_waypoint is None:
            wp_xy = np.array([0, 0], dtype=np.float32)
        else:
            wp_loc = self.next_waypoint.transform.location
            wp_xy = np.array([wp_loc.x, wp_loc.y], dtype=np.float32)

        # Speed
        speed = np.array([self.get_vehicle_speed()], dtype=np.float32)

        return {
            "segmentation": seg_img,
            "dist_center": dist_center,
            "gps_next_waypoint": wp_xy,
            "gps_own": gps_own,
            "speed": speed
        }

    def _show_image(self, seg_img):
        """
        Zeigt das Segmentationsbild per OpenCV-Fenster (480x640).
        """
        gray = (seg_img[..., 0] * 255).astype(np.uint8)  # (480,640)
        cv2.imshow("CARLA Semantic Segmentation", gray)
        cv2.waitKey(1)

    def render(self, mode="human"):
        pass

    def close(self):
        self._clear_actors()
        self.world.apply_settings(self.original_settings)
        cv2.destroyAllWindows()
