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
    Berechnet die laterale Distanz (Seitlicher Versatz) des Fahrzeugs zum 'center' 
    des Waypoints. Ignoriert z-Höhenunterschiede.
    """
    veh_loc = vehicle_transform.location
    wp_loc = waypoint_transform.location

    # 1) Vektor vom Waypoint-Zentrum zum Fahrzeug
    dx = veh_loc.x - wp_loc.x
    dy = veh_loc.y - wp_loc.y

    # 2) Vorwärtsrichtung am Waypoint
    forward = waypoint_transform.get_forward_vector()
    fx, fy = forward.x, forward.y

    # 3) "Quer"-Abstand über 2D-Kreuzprodukt 
    #    cross = (dx, dy) x (fx, fy) = dx*fy - dy*fx
    cross_val = dx * fy - dy * fx
    # Betragsmäßig könnte man die Norm durch forward-Norm teilen, 
    # aber forward sollte hier normiert sein => ~1.0
    return cross_val  # positive oder negative Werte sind möglich.

def is_waypoint_behind(vehicle_transform, waypoint_transform):
    """
    Prüft, ob der Waypoint "hinter" dem Fahrzeug liegt.
    Das geht z.B. über das Skalarprodukt zwischen Fahrzeug-Vorwärtsrichtung
    und dem Vektor Fahrzeug->Waypoint.
    Ist das Ergebnis < 0, liegt der Waypoint hinter dem Fahrzeug.
    """
    veh_loc = vehicle_transform.location
    wp_loc = waypoint_transform.location

    forward = vehicle_transform.get_forward_vector()
    to_waypoint = wp_loc - veh_loc

    # Dot Product in 3D
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
    """
    def __init__(self, host='localhost', port=2000, display=True):
        super().__init__()
        self.host = host
        self.port = port
        self.display = display  # Wenn True, zeigen wir das Kamera-Bild per cv2 an.

        # Verbinde mit CARLA
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)

        # Lade eine Beispielstadt (z.B. Town01). Achtung: Dauert manchmal etwas.
        self.world = self.client.load_world('Town01')
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()

        # Synchronous mode einstellen
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

        # Thread-Lock für Kameradaten
        self.image_lock = threading.Lock()
        self.camera_image = None

        # Actor placeholders
        self.vehicle = None
        self.collision_sensor = None
        self.camera_sensor = None

        # Wir halten den "nächsten Waypoint" (carla.Waypoint) fest
        self.next_waypoint = None

        # ACTION SPACE: [steer, throttle] in [-0.5, +0.5]
        # Wir klippen zur Sicherheit im Code noch einmal.
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(2,), dtype=np.float32)

        # OBSERVATION SPACE: Dict:
        #   segmentation: (480,640,1)
        #   dist_center: (1,)
        #   gps_next_waypoint: (2,)
        #   gps_own: (2,)
        #   speed: (1,)
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

        self.reset()

    # ---------------------------------
    # Hilfsfunktionen
    # ---------------------------------
    def _init_vehicle_sensors(self):
        # Fahrzeug spawnen
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
            labels /= 22.0  # Normierung auf [0,1], 22 ist eine typische Obergrenze in CARLA

            labels = np.expand_dims(labels, axis=-1)  # (480,640) => (480,640,1)
            with self.image_lock:
                self.camera_image = labels

        self.camera_sensor = CameraSensor(
            self.vehicle, self.blueprint_library, self.world, camera_callback
        )

        # Warmlaufen lassen
        for _ in range(10):
            self.world.tick()

        # Initialisiere den "nächsten Waypoint"
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
        """Setzt self.next_waypoint auf einen Waypoint in Fahrtrichtung oder
        bei Kreuzung zufällig (Rechts/Links/Geradeaus)."""
        if not self.vehicle:
            return

        veh_transform = self.vehicle.get_transform()
        current_wp = self.map.get_waypoint(
            veh_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        # Wir gehen z.B. 5 Meter weiter. (In Carla kann man next(1.0), next(5.0), etc. aufrufen.)
        possible_next = current_wp.next(5.0)

        if len(possible_next) == 0:
            # Falls es keinen nächsten Waypoint gibt, picken wir einfach neu
            return

        # Bei Kreuzungen liefert next() meist mehrere Optionen. Wenn wir mehr als 1 haben => random Wahl:
        self.next_waypoint = random.choice(possible_next)

        # Check: Wenn das next_waypoint in einer Kreuzung liegt, kann es sein,
        # dass es weitere Verzweigungen gibt => das könnte man weiter ausbauen,
        # indem man an dem next_waypoint nochmals .next(5.0) abfragt usw.
        # Für ein Minimalbeispiel bleiben wir so.

    def _remove_all_vehicles(self):
        """Löscht alle Fahrzeuge in der Welt."""
        actors = self.world.get_actors().filter("*vehicle*")
        for a in actors:
            a.destroy()

    def get_vehicle_speed(self):
        """Geschwindigkeit in km/h oder m/s? Hier mal m/s."""
        if not self.vehicle:
            return 0.0
        vel = self.vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    # ---------------------------------
    # Gym-Methoden
    # ---------------------------------
    def reset(self):
        # Alte Actoren entfernen
        self._clear_actors()

        # Neu spawnen
        self._init_vehicle_sensors()

        return self._get_obs()

    def step(self, action):
        # Action = [steer, gas], jeweils in [-0.5, 0.5]
        steer = float(clamp(action[0], -0.5, 0.5))
        throttle = float(clamp(action[1], -0.5, 0.5))

        # Normalerweise ist Throttle in [0,1]. Falls du 0.5 als Max willst,
        # dann interpretieren wir throttle = 0.5 => 50% Gas
        # also hier linear in [0,0.5].
        # (Man kann es aber auch direkt in [0,1] skalieren.)
        throttle = (throttle + 0.5)  # nun in [0, 1], falls wir -0.5 .. +0.5 hatten
        throttle = clamp(throttle, 0.0, 0.5)  # Safety

        control = carla.VehicleControl()
        control.steer = steer  # [-0.5, 0.5]
        control.throttle = throttle
        self.vehicle.apply_control(control)

        # Einen Tick simulieren
        self.world.tick()

        # Reward, Done, Info berechnen
        reward, done, info = self._compute_reward_done_info()

        # Falls Waypoint hinter uns => Reset Logik
        if is_waypoint_behind(self.vehicle.get_transform(), self.next_waypoint.transform):
            # Lösche alle Fahrzeuge und wähle neuen Spawnpoint
            self._remove_all_vehicles()
            done = True
            info["waypoint_behind"] = True

        obs = self._get_obs()
        return obs, reward, done, info

    def _compute_reward_done_info(self):
        """Berechnet Reward und Done."""
        info = {}
        done = False
        reward = 0.0

        # 1) Kollision
        if len(self.collision_sensor.get_history()) > 0:
            reward = -1.0
            done = True
            info["collision"] = True
            return reward, done, info  # Terminiert sofort

        # 2) Distanz zur Fahrbahnmitte -> in [-1, +1]
        #    Je näher bei 0 => desto besser
        lateral_offset = compute_lateral_offset(
            self.vehicle.get_transform(),
            self.map.get_waypoint(self.vehicle.get_transform().location).transform
        )
        # z.B. wir nehmen einen MaxOffset von 2.0m an. Alles darüber => -1
        max_offset = 2.0
        offset_magnitude = abs(lateral_offset)
        if offset_magnitude >= max_offset:
            dist_center_reward = -0.5
        else:
            # linear abnehmend von 0 -> -0.5
            dist_center_reward = 0.5 * (1.0 - offset_magnitude / max_offset)

        # 3) Geschwindigkeit (m/s)
        speed = self.get_vehicle_speed()
        # Bestrafe, wenn speed sehr klein: heading error
        # z.B. speed < 0.1 => -0.3
        if speed < 0.1:
            speed_reward = -0.3
        else:
            # z.B. normalisieren wir speed in [0..10] => (0..10) -> (0..0.5)
            # d.h. ab 10 m/s (36 km/h) capped
            capped_speed = min(speed, 10.0)
            speed_reward = 0.5 * (capped_speed / 10.0)

        # Summe
        reward = dist_center_reward + speed_reward
        # Wir clampen am Ende hart auf [-1,1]
        reward = clamp(reward, -1.0, 1.0)
        return reward, done, info

    def _get_obs(self):
        """
        Gibt Dictionary-Observation zurück:
          {
             "segmentation": ... (480,640,1),
             "dist_center": [float],
             "gps_next_waypoint": [x_wp, y_wp],
             "gps_own": [x_veh, y_veh],
             "speed": [float],
          }
        """
        with self.image_lock:
            if self.camera_image is None:
                seg_img = np.zeros((480, 640, 1), dtype=np.float32)
            else:
                seg_img = self.camera_image.copy()

        # Falls wir jedes Mal das Bild anzeigen wollen:
        if self.display:
            self._show_image(seg_img)

        # Distanz zur Mitte (lateraler offset)
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
            # Falls None (Fehlerfall) => (0,0)
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
        Zeigt das Segmentationsbild per OpenCV-Fenster.
        seg_img: shape (480,640,1) in [0,1].
        """
        gray = (seg_img[..., 0] * 255).astype(np.uint8)  # (480,640)
        cv2.imshow("CARLA Semantic Segmentation", gray)
        cv2.waitKey(1)

    def render(self, mode="human"):
        """Hier nichts extra nötig, da wir _show_image() schon beim _get_obs() aufrufen."""
        pass

    def close(self):
        self._clear_actors()
        self.world.apply_settings(self.original_settings)
        cv2.destroyAllWindows()
