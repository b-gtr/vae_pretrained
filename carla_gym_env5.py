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
    Berechnet die laterale Distanz (seitlicher Versatz) des Fahrzeugs
    zum Center des Waypoints. Ignoriert z-Höhenunterschiede.
    """
    veh_loc = vehicle_transform.location
    wp_loc = waypoint_transform.location

    dx = veh_loc.x - wp_loc.x
    dy = veh_loc.y - wp_loc.y

    forward = waypoint_transform.get_forward_vector()
    fx, fy = forward.x, forward.y

    # 2D-Kreuzprodukt => dx*fy - dy*fx
    cross_val = dx * fy - dy * fx
    return cross_val  # positive oder negative Werte möglich

def is_waypoint_behind(vehicle_transform, waypoint_transform):
    """
    Prüft, ob der Waypoint "hinter" dem Fahrzeug liegt, via Skalarprodukt
    zwischen Vorwärtsrichtung und Vektor Fahrzeug->Waypoint (< 0 => hinten).
    """
    veh_loc = vehicle_transform.location
    wp_loc = waypoint_transform.location

    forward = vehicle_transform.get_forward_vector()
    to_waypoint = wp_loc - veh_loc

    dot = (forward.x * to_waypoint.x
           + forward.y * to_waypoint.y
           + forward.z * to_waypoint.z)
    return dot < 0.0

# ---------------------------------
# Welt->Bild Projektion (Birdview)
# ---------------------------------
def world_to_birdview(world_loc, sensor_transform, image_width, image_height, meters_per_pixel=0.2):
    """
    Vereinfachte Projektion von Weltkoordinaten in die Bildkoordinaten (Birdview-Kamera).
    Annahme: pitch=-90°, yaw=0° => Kamera blickt nach unten. 
             +X -> rechts, +Y -> "oben" im Bild.
    Args:
        world_loc (carla.Location): Weltposition, die projiziert werden soll.
        sensor_transform (carla.Transform): Position/Rotation der Kamera.
        image_width (int), image_height (int)
        meters_per_pixel (float): Skalierung (Zoom) => 0.2 => 1 Pixel = 0.2 m
    Returns:
        (u, v) in Bildkoordinaten oder None, wenn es außerhalb liegt.
    """
    sx, sy, _sz = sensor_transform.location.x, sensor_transform.location.y, sensor_transform.location.z

    # dx, dy = relative Position in der Horizontalebene
    dx = world_loc.x - sx
    dy = world_loc.y - sy

    # +dx => rechts, +dy => oben
    u = image_width / 2 + dx / meters_per_pixel
    v = image_height / 2 - dy / meters_per_pixel

    u = int(round(u))
    v = int(round(v))

    if 0 <= u < image_width and 0 <= v < image_height:
        return (u, v)
    else:
        return None


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
        
        # Semantische Kamera mit Birdview (Top-Down)
        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')

        # Kamera nur 12 m über dem Auto => näher am Fahrzeug
        self.transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=12.0),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
        )
        self.sensor = world.spawn_actor(camera_bp, self.transform, attach_to=vehicle)
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
    Environment mit Birdview-Semantic-Segmentation. 
    Wir färben die "Ziel-Lane" (basierend auf self.next_waypoint) in der Birdview ein.
    Und ACHTUNG: Nach reset() wird next_waypoint sicher erneuert.
    """
    def __init__(self, host='localhost', port=2000, display=True):
        super().__init__()
        self.host = host
        self.port = port
        self.display = display  # Zeige Kamerabild per OpenCV an

        # Verbinde mit CARLA
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)

        # Lade eine Beispiel-Stadt
        self.world = self.client.load_world('Town01')
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()

        # Synchronous Mode
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

        # Lock für die Kamera
        self.image_lock = threading.Lock()
        self.camera_image = None

        # Actoren
        self.vehicle = None
        self.collision_sensor = None
        self.camera_sensor = None

        # Nächster Waypoint
        self.next_waypoint = None

        # 3 Sekunden warten nach Reset => 3 / 0.05 = 60 Ticks
        self.wait_steps = 0
        self.wait_steps_total = int(3.0 / settings.fixed_delta_seconds)

        # Action Space
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(2,), dtype=np.float32)

        # Observation Space
        self.observation_space = spaces.Dict({
            "segmentation": spaces.Box(low=0.0, high=1.0, shape=(480, 640, 1), dtype=np.float32),
            "dist_center": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "gps_next_waypoint": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "gps_own": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "speed": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
        })

        # Start
        self.reset()

    # ---------------------------------
    # Setup / Teardown
    # ---------------------------------
    def _init_vehicle_sensors(self):
        # Fahrzeug spawnen
        vehicle_bp = self.blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
        spawn_point = random.choice(self.spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Kollision
        self.collision_sensor = CollisionSensor(self.vehicle, self.blueprint_library, self.world)

        # Kamera-Callback
        def camera_callback(image):
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))  # => (480,640,4)

            # Red-Kanal in [0..22], dann Normierung => [0..1]
            labels = array[:, :, 2].astype(np.float32) / 22.0
            labeled_img = labels.copy()

            # Lane einfärben
            self._highlight_lane(labeled_img)

            labeled_img = np.expand_dims(labeled_img, axis=-1)
            with self.image_lock:
                self.camera_image = labeled_img

        self.camera_sensor = CameraSensor(
            self.vehicle, self.blueprint_library, self.world, camera_callback
        )

        # Erstmal ein paar Ticks
        for _ in range(10):
            self.world.tick()

        # Neuen "nächsten" Waypoint initial setzen
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

    # ---------------------------------
    # Lane-Highlighting
    # ---------------------------------
    def _highlight_lane(self, labeled_img):
        """
        Färbt einen Abschnitt der "Ziel-Lane" ein, indem wir Waypoints abfragen
        und in der Birdview manuell label=0.8 setzen.
        """
        if self.next_waypoint is None or self.camera_sensor is None:
            return

        sensor_tf = self.camera_sensor.sensor.get_transform()
        lane_id = self.next_waypoint.lane_id
        road_id = self.next_waypoint.road_id
        lane_width = self.next_waypoint.lane_width

        step = 2.0
        max_dist = 50.0
        current_wp = self.next_waypoint
        lane_waypoints = []
        dist_so_far = 0.0

        # Durch die Spur ~50m nach vorne samplen
        while dist_so_far < max_dist:
            lane_waypoints.append(current_wp)
            nxt_list = current_wp.next(step)
            found_wp = None
            for cand in nxt_list:
                if cand.road_id == road_id and cand.lane_id == lane_id:
                    found_wp = cand
                    break
            if not found_wp:
                break
            current_wp = found_wp
            dist_so_far += step

        # Querschnitt an jedem waypoint einfärben
        lateral_samples = np.linspace(-lane_width/2, lane_width/2, num=7)

        for wp in lane_waypoints:
            base_tf = wp.transform
            right_vec = base_tf.get_right_vector()
            base_x = base_tf.location.x
            base_y = base_tf.location.y

            for lat in lateral_samples:
                px = base_x + right_vec.x * lat
                py = base_y + right_vec.y * lat
                uv = world_to_birdview(
                    carla.Location(x=px, y=py, z=base_tf.location.z),
                    sensor_tf,
                    image_width=640,
                    image_height=480,
                    meters_per_pixel=0.2
                )
                if uv is None:
                    continue
                (u, v) = uv
                labeled_img[v, u] = 0.8  # "extra-hell"

    # ---------------------------------
    # Waypoint-Picking
    # ---------------------------------
    def _pick_next_waypoint(self):
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
            self.next_waypoint = None
            return
        self.next_waypoint = random.choice(possible_next)

    # ---------------------------------
    # Gym: reset / step
    # ---------------------------------
    def reset(self):
        # Wichtig: Bei Reset den alten Waypoint verwerfen!
        self._clear_actors()
        self.next_waypoint = None

        valid_spawn_found = False
        while not valid_spawn_found:
            self._init_vehicle_sensors()
            if self.next_waypoint is None:
                # Kein Waypoint bekommen => nochmal
                self._clear_actors()
                self.next_waypoint = None
                continue

            # Ist dieser Waypoint hinterm Fahrzeug?
            if is_waypoint_behind(self.vehicle.get_transform(), self.next_waypoint.transform):
                print("Waypoint ist direkt hinter dem Fahrzeug. Neuer Versuch ...")
                self._clear_actors()
                self.next_waypoint = None
            else:
                valid_spawn_found = True

        # Jetzt 3 Sek. warten
        self.wait_steps = self.wait_steps_total
        return self._get_obs()

    def step(self, action):
        # Während wait_steps > 0 => ignorieren
        if self.wait_steps > 0:
            self.wait_steps -= 1
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.vehicle.apply_control(control)
            self.world.tick()
            obs = self._get_obs()
            return obs, 0.0, False, {}

        # Normale Steuerung
        steer = float(clamp(action[0], -0.5, 0.5))
        throttle = float(clamp(action[1], -0.5, 0.5))
        throttle = (throttle + 0.5)
        throttle = clamp(throttle, 0.0, 0.5)

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        self.vehicle.apply_control(control)
        self.world.tick()

        reward, done, info = self._compute_reward_done_info()

        # Waypoint updaten, falls wir auf richtiger Spur & nah genug
        if not done and self.next_waypoint is not None:
            current_wp = self.map.get_waypoint(
                self.vehicle.get_transform().location, lane_type=carla.LaneType.Any
            )
            if current_wp.lane_id == self.next_waypoint.lane_id and current_wp.lane_type == carla.LaneType.Driving:
                dist = distance_2d(
                    vector_2d(self.vehicle.get_transform().location),
                    vector_2d(self.next_waypoint.transform.location)
                )
                if dist < 2.0:
                    print(f"Naher Waypoint (dist={dist:.2f}). Neuer Waypoint.")
                    self._pick_next_waypoint()
                elif is_waypoint_behind(self.vehicle.get_transform(), self.next_waypoint.transform):
                    print("Waypoint hinter uns. Neuer Waypoint.")
                    self._pick_next_waypoint()

        obs = self._get_obs()
        return obs, reward, done, info

    # ---------------------------------
    # Reward / Done
    # ---------------------------------
    def _compute_reward_done_info(self):
        info = {}
        done = False
        reward = 0.0

        # Kollision => Episode zu Ende
        if len(self.collision_sensor.get_history()) > 0:
            print(">>> Kollision!")
            reward = -1.0
            done = True
            info["collision"] = True
            return reward, done, info

        # Prüfe LaneType
        current_wp = self.map.get_waypoint(
            self.vehicle.get_transform().location,
            lane_type=carla.LaneType.Any
        )
        if current_wp.lane_type != carla.LaneType.Driving:
            print(">>> Off-Lane (Gehweg/Bordstein). Abbruch!")
            reward = -1.0
            done = True
            info["off_lane"] = True
            return reward, done, info

        # Lane-Mismatch
        mismatch = False
        if self.next_waypoint and (current_wp.lane_id != self.next_waypoint.lane_id):
            mismatch = True

        # Spurhaltung
        offset = abs(compute_lateral_offset(self.vehicle.get_transform(), current_wp.transform))
        max_offset = 1.0
        if offset >= max_offset:
            print(">>> Zu weit von der Spurmitte.")
            reward = -1.0
            done = True
            info["off_center"] = True
            return reward, done, info

        if mismatch:
            dist_center_reward = -0.5
        else:
            dist_center_reward = 0.5 * (1.0 - offset / max_offset)

        # Geschwindigkeit
        speed = self.get_vehicle_speed()
        if speed < 0.1:
            speed_reward = -0.3
        else:
            capped_speed = min(speed, 10.0)
            if mismatch:
                speed_reward = 0.0
            else:
                speed_reward = 0.5 * (capped_speed / 10.0)

        reward = dist_center_reward + speed_reward
        reward = clamp(reward, -1.0, 1.0)
        return reward, done, info

    # ---------------------------------
    # Observation
    # ---------------------------------
    def _get_obs(self):
        with self.image_lock:
            if self.camera_image is None:
                seg_img = np.zeros((480,640,1), dtype=np.float32)
            else:
                seg_img = self.camera_image.copy()

        if self.display:
            self._show_image(seg_img)

        # Distanz zur Mitte
        current_wp = self.map.get_waypoint(
            self.vehicle.get_transform().location,
            lane_type=carla.LaneType.Driving
        )
        if current_wp is not None:
            lateral_offset = compute_lateral_offset(
                self.vehicle.get_transform(),
                current_wp.transform
            )
        else:
            lateral_offset = 0.0

        dist_center = np.array([lateral_offset], dtype=np.float32)

        # GPS (x,y) Fahrzeug
        veh_loc = self.vehicle.get_transform().location
        gps_own = np.array([veh_loc.x, veh_loc.y], dtype=np.float32)

        # GPS (x,y) nächster Waypoint
        if self.next_waypoint is None:
            wp_xy = np.array([0.0, 0.0], dtype=np.float32)
        else:
            wp_loc = self.next_waypoint.transform.location
            wp_xy = np.array([wp_loc.x, wp_loc.y], dtype=np.float32)

        speed = np.array([self.get_vehicle_speed()], dtype=np.float32)

        return {
            "segmentation": seg_img,
            "dist_center": dist_center,
            "gps_next_waypoint": wp_xy,
            "gps_own": gps_own,
            "speed": speed
        }

    def _show_image(self, seg_img):
        gray = (seg_img[..., 0] * 255).astype(np.uint8)
        cv2.imshow("CARLA Semantic Segmentation (Highlighted Lane)", gray)
        cv2.waitKey(1)

    def render(self, mode="human"):
        pass

    def close(self):
        self._clear_actors()
        self.world.apply_settings(self.original_settings)
        cv2.destroyAllWindows()
