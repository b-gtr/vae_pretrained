import gym
from gym import spaces
import numpy as np
import pygame
import carla
import math
import threading


class Sensor:
    """
    Abstract Sensor class. Subclasses must implement `listen()` and set up their callback.
    """
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
    An example Gym environment wrapping CARLA.
    This environment:
      - Spawns a single vehicle at a fixed or random spawn point.
      - Attaches a semantic segmentation camera, collision, lane invasion, and GNSS sensors.
      - Expects a continuous action (steer, throttle).
      - Returns a simple reward based on collision, staying near lane center, etc.
      - Renders the latest camera frame using PyGame if `render_mode='human'`.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, host='localhost', port=2000, render_mode='human'):
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
        self.agent_image = None  # semantic segmentation image stored as float in [0,1]
        self.render_mode = render_mode

        # PyGame
        self.display = (self.render_mode == 'human')
        if self.display:
            pygame.init()
            self.screen = pygame.display.set_mode((640, 480))
            pygame.display.set_caption("CARLA Semantic Segmentation")
            self.clock = pygame.time.Clock()

        # Define action and observation space for Stable Baselines3
        # Example:
        #   action = [steer, throttle], each in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Example observation space: semantic segmentation image
        # shape = (480, 640, 1), values in [0,1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(480, 640, 1),
            dtype=np.float32
        )

        self.reset()

    def _init_actors(self):
        """
        Internal method to (re)spawn the vehicle and attach sensors.
        """
        # Destroy old sensors/vehicle if they exist
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
        into a [0,1]-normalized single-channel array.
        """
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        # Extract the semantic label from the red channel
        labels = array[:, :, 2].astype(np.float32)

        # Normalize to [0,1] by dividing by the maximum label (22.0 is CARLA's default label range up to 22).
        labels /= 22.0

        with self.image_lock:
            # Store as (480, 640, 1) so it matches the observation space shape
            self.agent_image = np.expand_dims(labels, axis=-1)

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
        # Parse action
        steer = float(action[0])  # in [-1,1]
        throttle = float(action[1])  # in [-1,1]
        # Example: scale throttle from [-1,1] to [0,1]
        throttle = 0.5 * (throttle + 1.0)

        # Apply control
        control = carla.VehicleControl()
        control.steer = np.clip(steer, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        self.vehicle.apply_control(control)

        # Tick the world
        self.world.tick()

        # Compute reward (placeholder example)
        reward = 0.0
        done = False
        info = {}

        # Collision penalty
        if len(self.collision_sensor.get_history()) > 0:
            reward -= 50.0
            done = True
            info['collision'] = True

        # Lane invasion check (just an example)
        if len(self.lane_invasion_sensor.get_history()) > 0:
            reward -= 10.0
            # Not necessarily done, but you can decide based on your scenario

        # A small reward for "moving forward" (using speed)
        speed = self.get_vehicle_speed()
        reward += speed * 0.1  # scale factor

        # Example "max steps" cut-off could also be used
        # if self.current_step >= self.max_steps:
        #     done = True
        #     info['TimeLimit.truncated'] = True

        obs = self._get_observation()
        return obs, reward, done, info

    def _get_observation(self):
        """
        Returns the latest camera observation if available, else a zero array.
        """
        with self.image_lock:
            if self.agent_image is None:
                # Return zeros if no image yet
                return np.zeros((480, 640, 1), dtype=np.float32)
            else:
                return self.agent_image.copy()

    def render(self, mode='human'):
        """
        Renders the semantic segmentation camera feed via PyGame.
        """
        if not self.display or self.agent_image is None:
            return

        # Convert the single channel [0,1] image to a 3-channel [0,255] image for PyGame
        img_3ch = np.repeat(self.agent_image, 3, axis=-1) * 255
        img_3ch = img_3ch.astype(np.uint8)

        surf = pygame.surfarray.make_surface(img_3ch.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

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
