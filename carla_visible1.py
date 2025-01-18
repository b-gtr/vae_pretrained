import gym
from gym import spaces
import numpy as np
import pygame
import carla
import math
import threading

class CollisionSensor:
    def __init__(self, vehicle, blueprint_library, world):
        self.vehicle = vehicle
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.history = []

    def listen(self):
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

    def listen(self):
        self.sensor.listen(self.callback)

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()


class CarlaGymEnv(gym.Env):
    """
    An environment using a single‐channel semantic‐segmentation camera in (480,640,1).
    This shape is recognized by SB3's NatureCNN as a grayscale image.
    """

    def __init__(self, host='localhost', port=2000, display=True):
        super().__init__()
        self.host = host
        self.port = port
        self.display = display

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
        self.camera_sensor = None

        # Thread lock to protect camera data
        self.image_lock = threading.Lock()
        # We'll store camera data in shape (480, 640, 1).
        self.agent_image = None

        # PyGame for rendering
        if self.display:
            pygame.init()
            self.screen = pygame.display.set_mode((640, 480))
            pygame.display.set_caption("CARLA Semantic Segmentation")
            self.clock = pygame.time.Clock()

        # ACTION SPACE: 2D continuous [steer, throttle] in [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # OBSERVATION SPACE: (480, 640, 1), channel-last
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(480, 640, 1),
            dtype=np.float32
        )

        self.reset()

    def _init_actors(self):
        # Clear old actors if any
        self._clear_actors()

        # Spawn a vehicle
        vehicle_bp = self.blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
        spawn_point = np.random.choice(self.spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Collision sensor
        self.collision_sensor = CollisionSensor(self.vehicle, self.blueprint_library, self.world)
        self.collision_sensor.listen()

        # Camera sensor
        def camera_callback(image):
            """
            Convert CARLA semantic segmentation image to shape (480,640,1) in [0,1].
            """
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))  # => (480,640,4)

            # Red channel has the semantic labels
            labels = array[:, :, 2].astype(np.float32)
            # Normalized to [0,1], 22 is the default max label in Carla
            labels /= 22.0

            # Expand to channel-last: (480,640) => (480,640,1)
            labels = np.expand_dims(labels, axis=-1)

            with self.image_lock:
                self.agent_image = labels  # shape (480,640,1)

        self.camera_sensor = CameraSensor(self.vehicle, self.blueprint_library, self.world, camera_callback)
        self.camera_sensor.listen()

        # Let the sensors warm up
        for _ in range(10):
            self.world.tick()

    def _clear_actors(self):
        if self.camera_sensor:
            self.camera_sensor.destroy()
            self.camera_sensor = None
        if self.collision_sensor:
            self.collision_sensor.destroy()
            self.collision_sensor = None
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None
        self.agent_image = None

    def reset(self):
        self._init_actors()
        return self._get_obs()

    def step(self, action):
        steer, throttle = float(action[0]), float(action[1])
        # Scale throttle from [-1,1] => [0,1]
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

        # Collision check
        if len(self.collision_sensor.get_history()) > 0:
            reward -= 50.0
            done = True
            info["collision"] = True

        # Speed-based reward
        speed = self.get_vehicle_speed()
        reward += speed * 0.1

        obs = self._get_obs()
        return obs, reward, done, info

    def _get_obs(self):
        with self.image_lock:
            if self.agent_image is None:
                return np.zeros((480, 640, 1), dtype=np.float32)
            else:
                return self.agent_image.copy()

    def render(self):
        if not self.display or self.agent_image is None:
            return

        # shape: (480,640,1)
        gray = (self.agent_image[..., 0] * 255).astype(np.uint8)  # (480,640)
        rgb = np.stack([gray, gray, gray], axis=-1)  # (480,640,3)

        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(20)

    def close(self):
        if self.display:
            pygame.quit()

        # restore settings
        self.world.apply_settings(self.original_settings)
        self._clear_actors()

    def get_vehicle_speed(self):
        if not self.vehicle:
            return 0.0
        vel = self.vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
