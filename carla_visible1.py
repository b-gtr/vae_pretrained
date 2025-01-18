import gym
import numpy as np
import pygame
import carla
import math
import threading
from gym import spaces

class CollisionSensor:
    def __init__(self, vehicle, blueprint_library, world):
        self.vehicle = vehicle
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.history = []

    def listen(self):
        self.sensor.listen(lambda event: self.history.append(event))

    def destroy(self):
        if self.sensor:
            self.sensor.destroy()

    def get_history(self):
        return self.history


class LaneInvasionSensor:
    def __init__(self, vehicle, blueprint_library, world):
        self.vehicle = vehicle
        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)
        self.history = []

    def listen(self):
        self.sensor.listen(lambda event: self.history.append(event))

    def destroy(self):
        if self.sensor:
            self.sensor.destroy()

    def get_history(self):
        return self.history


class CameraSensor:
    def __init__(self, vehicle, blueprint_library, world, image_callback):
        self.vehicle = vehicle
        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')
        transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.sensor = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
        self.callback = image_callback

    def listen(self):
        self.sensor.listen(self.callback)

    def destroy(self):
        if self.sensor:
            self.sensor.destroy()


class CarlaGymEnv(gym.Env):
    """
    CARLA environment returning a semantic segmentation camera image in (480,640,1).
    The image is stored as uint8 in [0..255].

    We adopt the newer Gym/Gymnasium style, with `render_mode` in the constructor.
    `render_mode="human"` => PyGame window, "rgb_array" => returns np.array in render().
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20
    }

    def __init__(self, host='localhost', port=2000, render_mode=None):
        super().__init__()
        self.host = host
        self.port = port
        self.render_mode = render_mode  # could be None, "human", or "rgb_array"

        # Connect to CARLA
        self.client = carla.Client(self.host, self.port)
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
        self.camera_sensor = None

        # Thread lock for camera data
        self.image_lock = threading.Lock()
        # We'll store the camera data in shape (480,640,1), uint8
        self.agent_image = None

        # If user wants a live window: init PyGame
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((640, 480))
            pygame.display.set_caption("CARLA Semantic Segmentation")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None

        # Action space: [steer, throttle] in [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: single‐channel 8-bit image => [0..255], shape=(480,640,1)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(480, 640, 1),
            dtype=np.uint8
        )

        self.reset()

    def _init_actors(self):
        self._clear_actors()

        # Spawn the vehicle
        vehicle_bp = self.blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
        spawn_point = np.random.choice(self.spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Collision sensor
        self.collision_sensor = CollisionSensor(self.vehicle, self.blueprint_library, self.world)
        self.collision_sensor.listen()

        # Lane invasion sensor
        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self.blueprint_library, self.world)
        self.lane_invasion_sensor.listen()

        # Camera sensor
        def camera_callback(image):
            # Convert semantic segmentation to single‐channel [0..255]
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))  # => (480,640,4)

            # The "red" channel has semantic labels in [0..22], multiply => [0..255]
            labels = array[..., 2].astype(np.float32)
            labels *= (255.0 / 22.0)
            labels = np.clip(labels, 0, 255).astype(np.uint8)

            # Expand to (480,640,1) for single‐channel last
            labels = np.expand_dims(labels, axis=-1)

            with self.image_lock:
                self.agent_image = labels

        self.camera_sensor = CameraSensor(self.vehicle, self.blueprint_library, self.world, camera_callback)
        self.camera_sensor.listen()

        # Let sensors warm up
        for _ in range(10):
            self.world.tick()

    def _clear_actors(self):
        if self.camera_sensor:
            self.camera_sensor.destroy()
            self.camera_sensor = None
        if self.collision_sensor:
            self.collision_sensor.destroy()
            self.collision_sensor = None
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.destroy()
            self.lane_invasion_sensor = None
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None
        self.agent_image = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_actors()
        return self._get_obs()

    def step(self, action):
        steer, throttle = float(action[0]), float(action[1])
        # scale throttle from [-1,1] => [0,1]
        throttle = 0.5 * (throttle + 1.0)

        control = carla.VehicleControl()
        control.steer = np.clip(steer, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        self.vehicle.apply_control(control)

        self.world.tick()

        # Simple reward
        reward = 0.0
        done = False
        info = {}

        # Collision penalty
        if len(self.collision_sensor.get_history()) > 0:
            reward -= 50.0
            done = True
            info["collision"] = True

        # Lane invasion penalty
        if len(self.lane_invasion_sensor.get_history()) > 0:
            reward -= 10.0

        # Speed-based reward
        speed = self.get_vehicle_speed()
        reward += speed * 0.1

        obs = self._get_obs()
        return obs, reward, done, info

    def _get_obs(self):
        with self.image_lock:
            if self.agent_image is None:
                return np.zeros((480, 640, 1), dtype=np.uint8)
            return self.agent_image.copy()

    def render(self, mode=None):
        """
        Called by SB3 or Gym. We accept 'mode' so that we don't get 'unexpected keyword argument' errors.
        If render_mode == "human" => show PyGame window.
        If "rgb_array" => return np.array of the frame.
        """
        # If no mode is given, default to self.render_mode
        if mode is None:
            mode = self.render_mode

        if mode == "human":
            # Show live PyGame window
            if self.screen is not None and self.agent_image is not None:
                # agent_image is (480,640,1); replicate to (480,640,3) for a quick color image
                gray = self.agent_image[..., 0]
                rgb = np.stack([gray, gray, gray], axis=-1)
                surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                self.screen.blit(surf, (0, 0))
                pygame.display.flip()
                if self.clock:
                    self.clock.tick(self.metadata["render_fps"])
            return None
        elif mode == "rgb_array":
            # Return the latest frame as an array
            if self.agent_image is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            # Convert (480,640,1) => (480,640,3)
            gray = self.agent_image[..., 0]
            rgb = np.stack([gray, gray, gray], axis=-1)
            return rgb
        else:
            # mode=None or something else => do nothing
            return None

    def close(self):
        if self.render_mode == "human" and self.screen is not None:
            pygame.quit()
        self.world.apply_settings(self.original_settings)
        self._clear_actors()

    def get_vehicle_speed(self):
        if not self.vehicle:
            return 0.0
        vel = self.vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
