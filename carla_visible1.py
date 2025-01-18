import gym
from gym import spaces
import numpy as np
import pygame
import carla
import math
import threading

class Sensor:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.sensor = None
        self.history = []

    def listen(self):
        raise NotImplementedError

    def clear_history(self):
        self.history.clear()

    def destroy(self):
        if self.sensor is not None:
            try:
                self.sensor.destroy()
            except RuntimeError as e:
                print(f"Error destroying sensor: {e}")
        
    def get_history(self):
        return self.history


class CollisionSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world):
        super().__init__(vehicle)
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)

    def _on_collision(self, event):
        self.history.append(event)

    def listen(self):
        self.sensor.listen(self._on_collision)


class LaneInvasionSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world):
        super().__init__(vehicle)
        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)

    def _on_lane_invasion(self, event):
        self.history.append(event)

    def listen(self):
        self.sensor.listen(self._on_lane_invasion)


class GnssSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world):
        super().__init__(vehicle)
        gnss_bp = blueprint_library.find('sensor.other.gnss')
        self.sensor = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=self.vehicle)
        self.current_gnss = None

    def _on_gnss_event(self, event):
        self.current_gnss = event

    def listen(self):
        self.sensor.listen(self._on_gnss_event)
    
    def get_current_gnss(self):
        return self.current_gnss


class CameraSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world, image_callback):
        super().__init__(vehicle)
        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.image_callback = image_callback

    def listen(self):
        self.sensor.listen(self.image_callback)


class CarlaGymEnv(gym.Env):
    """
    A Gym environment wrapper for CARLA that spawns a vehicle with:
      - Collision sensor
      - Lane invasion sensor
      - GNSS sensor
      - Semantic segmentation camera

    The environment expects a 2D continuous action [steer, throttle] in [-1, 1].
    Observations are semantic segmentation images in channel-first format: (1, 480, 640), in [0, 1].
    
    Set display=True to open a PyGame window for rendering.
    """

    def __init__(self, host='localhost', port=2000, display=True):
        """
        :param host: CARLA server host
        :param port: CARLA server port
        :param display: bool, whether to open a PyGame window to display the camera feed
        """
        super(CarlaGymEnv, self).__init__()

        # Connect to CARLA
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town01')
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

        # Synchronous mode
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

        self.vehicle = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_sensor = None

        # Thread lock for image access
        self.image_lock = threading.Lock()
        # We'll store the camera data in a (1, 480, 640) array
        self.agent_image = None  

        # PyGame related
        self.display = display
        if self.display:
            pygame.init()
            self.screen = pygame.display.set_mode((640, 480))
            pygame.display.set_caption("CARLA Semantic Segmentation")
            self.clock = pygame.time.Clock()

        # Define action and observation spaces
        # Action: 2D continuous [steer, throttle]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: single-channel image with shape (1, 480, 640), values in [0,1].
        #   (channel-first format for Stable Baselines3).
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, 480, 640),
            dtype=np.float32
        )

        self.reset()

    def _init_actors(self):
        """
        Internal method to (re)spawn the vehicle and attach sensors.
        """
        self._clear_sensors()

        # Spawn vehicle at a random or fixed spawn point
        vehicle_bp = self.blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
        spawn_point = np.random.choice(self.spawn_points)  # random spawn
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Collision sensor
        self.collision_sensor = CollisionSensor(self.vehicle, self.blueprint_library, self.world)
        self.collision_sensor.listen()

        # Lane invasion sensor
        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self.blueprint_library, self.world)
        self.lane_invasion_sensor.listen()

        # GNSS sensor
        self.gnss_sensor = GnssSensor(self.vehicle, self.blueprint_library, self.world)
        self.gnss_sensor.listen()

        # Camera sensor
        self.camera_sensor = CameraSensor(self.vehicle, self.blueprint_library, self.world, self._on_camera_image)
        self.camera_sensor.listen()

        # Give the sensors some ticks to stabilize
        for _ in range(10):
            self.world.tick()

    def _on_camera_image(self, image):
        """
        Callback for the camera sensor. Converts the raw segmentation image
        into a [0,1]-normalized single-channel array in channel-first format.
        """
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        # Extract the semantic label from the red channel
        labels = array[:, :, 2].astype(np.float32)

        # Normalize to [0,1] by dividing by max label (22.0 is CARLA's default max label).
        labels /= 22.0

        # Currently shape is (480, 640).
        # Convert to channel-first format -> (1, 480, 640).
        labels = np.expand_dims(labels, axis=0)

        with self.image_lock:
            self.agent_image = labels  # shape: (1, 480, 640)

    def _clear_sensors(self):
        """
        Destroy existing sensors and vehicle.
        """
        if self.camera_sensor:
            self.camera_sensor.destroy()
            self.camera_sensor = None
        if self.collision_sensor:
            self.collision_sensor.destroy()
            self.collision_sensor = None
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.destroy()
            self.lane_invasion_sensor = None
        if self.gnss_sensor:
            self.gnss_sensor.destroy()
            self.gnss_sensor = None
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None
        self.agent_image = None

    def reset(self):
        """
        Gym reset. Respawns the vehicle and sensors, returns the initial observation.
        """
        self._init_actors()
        return self._get_observation()

    def step(self, action):
        """
        Apply action, perform a tick in the CARLA world,
        compute reward, and check for done.
        """
        steer = float(action[0])  # in [-1,1]
        throttle = float(action[1])  # in [-1,1]
        throttle = 0.5 * (throttle + 1.0)  # scale [-1,1] to [0,1]

        # Apply control
        control = carla.VehicleControl()
        control.steer = np.clip(steer, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        self.vehicle.apply_control(control)

        # Tick the world
        self.world.tick()

        # Compute reward (simple example)
        reward = 0.0
        done = False
        info = {}

        # Collision penalty
        if len(self.collision_sensor.get_history()) > 0:
            reward -= 50.0
            done = True
            info['collision'] = True

        # Lane invasion penalty
        if len(self.lane_invasion_sensor.get_history()) > 0:
            reward -= 10.0

        # Reward for moving forward (scaled by speed)
        speed = self.get_vehicle_speed()
        reward += speed * 0.1

        obs = self._get_observation()
        return obs, reward, done, info

    def _get_observation(self):
        """
        Returns the latest camera observation if available, else a zero array.
        Note that we keep the shape (1, 480, 640) for channel-first.
        """
        with self.image_lock:
            if self.agent_image is None:
                return np.zeros((1, 480, 640), dtype=np.float32)
            else:
                return self.agent_image.copy()

    def render(self):
        """
        Renders the semantic segmentation camera feed via PyGame if display=True.
        Note: PyGame expects (width, height), so we transpose the channel-first array.
        """
        if not self.display or self.agent_image is None:
            return

        # self.agent_image shape: (1, 480, 640)
        # Convert to (480, 640) for single-channel
        img_2d = self.agent_image[0] * 255.0  # back to [0,255]
        img_2d = img_2d.astype(np.uint8)

        # Convert single channel to 3-ch for PyGame display
        img_3ch = np.stack([img_2d, img_2d, img_2d], axis=-1)  # shape (480, 640, 3)

        surf = pygame.surfarray.make_surface(img_3ch.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(20)  # 20 FPS

    def close(self):
        """
        Cleanup upon closing the environment.
        """
        if self.display:
            pygame.quit()

        # Restore original CARLA settings
        self.world.apply_settings(self.original_settings)
        self._clear_sensors()

    def get_vehicle_speed(self):
        """
        Returns the speed of the vehicle in m/s.
        """
        if not self.vehicle:
            return 0.0
        vel = self.vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    def get_lane_center_and_offset(self):
        """
        Returns the lane center location and lateral offset of the vehicle from lane center.
        """
        if not self.vehicle:
            return None, None

        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        map_ = self.world.get_map()
        waypoint = map_.get_waypoint(vehicle_location, project_to_road=True)
        if not waypoint:
            return None, None

        lane_center = waypoint.transform.location
        dx = vehicle_location.x - lane_center.x
        dy = vehicle_location.y - lane_center.y

        lane_heading = math.radians(waypoint.transform.rotation.yaw)
        lane_direction = carla.Vector3D(math.cos(lane_heading), math.sin(lane_heading), 0)
        perpendicular_direction = carla.Vector3D(-lane_direction.y, lane_direction.x, 0)

        lateral_offset = dx * perpendicular_direction.x + dy * perpendicular_direction.y
        return lane_center, lateral_offset
