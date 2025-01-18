import gym
import pygame
import numpy as np
from gym import spaces
from gym.utils import seeding

import carla
import random
import time
import math

#########################
# CarlaGymEnv-Klasse
#########################

class CarlaGymEnv(gym.Env):
    """
    Carla Gym-Wrapper zum Trainieren mit SB3 (MultiInputLstmPolicy).
    Beobachtungen:
      1) Segmentierte Kamera: shape = (IMG_HEIGHT, IMG_WIDTH, 3)
      2) Array mit Werten:
         [dist_center, gps_waypoint_x, gps_waypoint_y, gps_ego_x, gps_ego_y, speed]

    Aktionen (kontinuierlich):
      - Gas (Throttle)  in [0.0,  0.5]
      - Lenken (Steering) in [-0.5, 0.5]

    Rewards (Beispielhaft):
      - +1 max für perfekte Spurhaltung
      -  negative Belohnung je größer Abweichung von Fahrbahnmitte
      -  Heading/Speed-Strafe, wenn zu langsam oder im falschen Winkel
      -  Kollision => Terminierung mit -1
      -  Alle Rewards werden gescaled in [-1,1]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        host="127.0.0.1",
        port=2000,
        img_width=128,
        img_height=128,
        seconds_per_episode=30.0,
        render_pygame=True
    ):
        super(CarlaGymEnv, self).__init__()

        self.host = host
        self.port = port
        self.client = None
        self.world = None
        self.vehicle = None

        # Bilddimensionen
        self.IMG_WIDTH = img_width
        self.IMG_HEIGHT = img_height

        # Wie lange ein Episode dauern darf
        self.seconds_per_episode = seconds_per_episode

        # Sensor-Handler
        self.camera_sensor = None
        self.front_camera = None  # wird Numpy-Array (H, W, 3) sein
        self.collision_sensor = None

        # Pygame-Rendering einschalten?
        self.render_pygame = render_pygame
        self._pygame_initialized = False

        # Pygame-Fenster
        if self.render_pygame:
            pygame.init()
            self.display = pygame.display.set_mode((self.IMG_WIDTH, self.IMG_HEIGHT))
            pygame.display.set_caption("CARLA Semantic Segmentation")
            self._pygame_initialized = True

        # Observation Space:
        #   1) Kamerabild: Box(0, 255, shape=(H,W,3))
        #   2) Array dist_center, gps_wp_x, gps_wp_y, gps_ego_x, gps_ego_y, speed
        #      => z.B. 6 floats, wir schätzen Bereiche hier nur grob ab.
        camera_space = spaces.Box(low=0, high=255, shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3), dtype=np.uint8)
        state_space = spaces.Box(
            low=np.array([-10.0, -1e5, -1e5, -1e5, -1e5, 0.0]),
            high=np.array([10.0, 1e5, 1e5, 1e5, 1e5, 300.0]),
            dtype=np.float32
        )

        # Für MultiInputLstmPolicy brauchen wir ein Dictionary
        self.observation_space = spaces.Dict(
            {
                "camera": camera_space,
                "state": state_space,
            }
        )

        # Action Space:
        #   Throttle in [0, 0.5]
        #   Steering in [-0.5, 0.5]
        # => Box(low=[0.0, -0.5], high=[0.5, 0.5])
        self.action_space = spaces.Box(
            low=np.array([0.0, -0.5], dtype=np.float32),
            high=np.array([0.5, 0.5], dtype=np.float32),
        )

        # Kollisions-Historie
        self.collision_history = []

        # Episode-Start-Time
        self.episode_start_time = None

        # Wegpunkt-Infos
        self.map = None
        self.current_waypoint = None
        self.next_waypoint = None

        # Zufallsgenerator für gym
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def reset(self):
        """
        Reset-Methode: verbindet mit CARLA, erstellt Welt und Fahrzeug neu.
        Wählt Spawn-Point, setzt Sensoren etc.
        """
        # Falls schon einmal Welt/Vehicle vorhanden -> zerstören
        self._cleanup_actors()

        # Client/Welt initialisieren
        if self.client is None:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        self.map = self.world.get_map()

        # Zufälligen Spawn-Punkt
        spawn_points = self.map.get_spawn_points()
        spawn_point = random.choice(spawn_points)

        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("model3")[0]  # z.B. Tesla Model 3
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if self.vehicle is None:
            # Falls besetzt -> nochmal probieren (minimaler Fallback)
            for _ in range(10):
                spawn_point = random.choice(spawn_points)
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if self.vehicle is not None:
                    break
            if self.vehicle is None:
                raise RuntimeError("Fahrzeug konnte nicht gespawnt werden.")

        # Kamera-Blueprint (Semantic Segmentation)
        seg_cam_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        seg_cam_bp.set_attribute("image_size_x", str(self.IMG_WIDTH))
        seg_cam_bp.set_attribute("image_size_y", str(self.IMG_HEIGHT))
        seg_cam_bp.set_attribute("fov", "100")

        # Kamera am Fahrzeug anbringen
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # leicht über dem Auto
        self.camera_sensor = self.world.spawn_actor(seg_cam_bp, cam_transform, attach_to=self.vehicle)

        # Callback registrieren
        self.camera_sensor.listen(lambda image: self._on_camera_image(image))

        # Collision Sensor
        col_bp = blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        # Kollisions-Historie leeren
        self.collision_history = []

        # Fahrzeug "anfahren"
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # Wegpunkt-Initialisierung
        self.current_waypoint = self.map.get_waypoint(self.vehicle.get_location())
        self.next_waypoint = self._get_random_next_waypoint(self.current_waypoint)

        # Startzeit der Episode
        self.episode_start_time = time.time()

        # Warten, bis erstes Kamerabild da ist
        self.front_camera = None
        start_wait = time.time()
        while self.front_camera is None:
            time.sleep(0.01)
            if time.time() - start_wait > 5.0:
                break

        # Beobachtung zurückgeben
        obs = self._get_observation()
        return obs

    def step(self, action):
        """
        Step-Funktion: Führe Aktion aus, berechne Reward, prüfe Termination.
        action = [throttle, steer]
        """
        throttle, steer = float(action[0]), float(action[1])

        # Kontrolle ans Fahrzeug senden
        control = carla.VehicleControl()
        control.throttle = max(0.0, min(0.5, throttle))  # clip
        control.steer = max(-0.5, min(0.5, steer))       # clip
        control.brake = 0.0
        self.vehicle.apply_control(control)

        # Geschw./Position/Abstand usw. berechnen
        distance_to_center, heading_error = self._calc_lane_center_data()

        # Reward design (vereinfacht):
        # 1) Spurhaltung: Je näher an der Mitte, desto besser
        # 2) Heading/Speed-Strafe
        # 3) Collision = done + -1
        # 4) minimaler Reward = -1, maximaler Reward = +1

        done = False
        reward = 0.0

        # Kollision?
        if len(self.collision_history) > 0:
            done = True
            reward = -1.0

        # Distance zur Mitte => normalisieren und belohnen
        # Hier z.B. lineare Abnahme. Distanz > 2.0 => starker Malus
        dist_penalty = min(abs(distance_to_center) / 2.0, 1.0)
        reward_center = 1.0 - dist_penalty  # [0..1]
        reward += 0.5 * reward_center  # Wichtung

        # Geschwindigkeit
        speed = self._get_speed()
        if speed < 0.1:
            # steht quasi -> Malus
            reward -= 0.3

        # Heading-Error: stark vereinfacht
        # Hier z.B. |heading_error| > 25 Grad => Malus
        if abs(heading_error) > 25.0:
            reward -= 0.2

        # Reward begrenzen
        reward = float(np.clip(reward, -1.0, 1.0))

        # Timeout => done
        if (time.time() - self.episode_start_time) > self.seconds_per_episode:
            done = True

        # Prüfen, ob Waypoint "hinter" uns liegt => neu spawnen
        if not self._waypoint_ahead_of_vehicle(self.next_waypoint):
            # Alle Vehicles löschen (bspw. NPCs)
            self._remove_all_vehicles()
            # Neues Spawn
            self.reset()
            done = True
            reward -= 0.3  # kleiner Malus, da wir "gescheitert" sind

        # Random Entscheidung an Kreuzungen
        if self.next_waypoint.is_intersection:
            # z.B. 1/3 gerade, 1/3 rechts, 1/3 links
            r = random.random()
            if r < 0.33:
                self.next_waypoint = self.next_waypoint.next_until_lane_end(1)[0]
            elif r < 0.66:
                self.next_waypoint = self.next_waypoint.get_right_lane()
            else:
                self.next_waypoint = self.next_waypoint.get_left_lane()
        else:
            # Normal weiter
            self.next_waypoint = self._get_random_next_waypoint(self.next_waypoint)

        obs = self._get_observation()

        return obs, reward, done, {}

    def render(self, mode="human"):
        """
        Optionales Rendering via Pygame.
        """
        if not self.render_pygame or not self._pygame_initialized:
            return

        if self.front_camera is not None:
            surface = pygame.surfarray.make_surface(self.front_camera.swapaxes(0, 1))
            self.display.blit(surface, (0, 0))
        pygame.display.flip()

    def close(self):
        """
        Aufräumen.
        """
        self._cleanup_actors()
        if self._pygame_initialized:
            pygame.quit()
        self.client = None

    #########################
    # Hilfsfunktionen
    #########################

    def _on_camera_image(self, image):
        """
        Callback für Kamera-Bild (Semantic Segmentation).
        Konvertieren in ein RGB-Array, in self.front_camera speichern.
        """
        # image: carla.Image => BGRA
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        # Die semantische Segmentierung liegt in 'G' (grüner Kanal) oder als 'class'-Index
        # Der Einfachheit halber tun wir so, als wäre es schon ein "farbiges" Labelbild.
        # Man kann hier eine eigene Farbpalette anwenden. Demo:
        rgb = array[:, :, :3]
        self.front_camera = rgb

    def _on_collision(self, event):
        """
        Callback für Kollisionen.
        """
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        # Wenn Intensität > Schwellwert => Kollision merken
        if intensity > 500:  # Threshold anpassen
            self.collision_history.append(event)

    def _get_speed(self):
        """
        Liefert Geschwindigkeit (km/h).
        """
        vel = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        return speed

    def _calc_lane_center_data(self):
        """
        Berechnet:
          - Distance zur Fahrbahnmitte (positiv/negativ je nach Seite, stark vereinfacht)
          - Heading-Fehler (in Grad)
        """
        waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True)
        if waypoint is None:
            return 0.0, 0.0

        # Distanz zur Center-Linie
        loc = self.vehicle.get_location()
        wp_loc = waypoint.transform.location
        vec_wp2veh = np.array([loc.x - wp_loc.x, loc.y - wp_loc.y])
        # Quer-/Normalenrichtung
        forward = waypoint.transform.get_forward_vector()
        # normal vector (perp):
        #  forward (fx, fy), normal ~ (fy, -fx)
        fx, fy = forward.x, forward.y
        perp = np.array([fy, -fx])
        dist_center = np.dot(vec_wp2veh, perp)

        # Heading Error
        veh_yaw = self.vehicle.get_transform().rotation.yaw
        wp_yaw = waypoint.transform.rotation.yaw
        heading_error = wp_yaw - veh_yaw
        heading_error = (heading_error + 180) % 360 - 180  # in [-180, 180]

        return dist_center, heading_error

    def _get_observation(self):
        """
        Fasst das Dict für MultiInputLstmPolicy zusammen.
         {
           'camera': np.array(H,W,3),
           'state': np.array([dist_center, gps_wp_x, gps_wp_y, ego_x, ego_y, speed])
         }
        """
        dist_center, _ = self._calc_lane_center_data()
        speed = self._get_speed()

        # next_waypoint Koordinaten
        if self.next_waypoint is not None:
            wp_loc = self.next_waypoint.transform.location
            gps_wp_x, gps_wp_y = wp_loc.x, wp_loc.y
        else:
            gps_wp_x, gps_wp_y = 0.0, 0.0

        # eigene Position
        loc = self.vehicle.get_location()
        ego_x, ego_y = loc.x, loc.y

        # Numerischer State
        state_array = np.array([dist_center, gps_wp_x, gps_wp_y, ego_x, ego_y, speed], dtype=np.float32)

        # Kamera
        if self.front_camera is None:
            # Platzhalter-Bild, falls noch nichts da
            camera_image = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 3), dtype=np.uint8)
        else:
            camera_image = self.front_camera

        return {
            "camera": camera_image,
            "state": state_array
        }

    def _get_random_next_waypoint(self, current_wp):
        """
        Hilfsfunktion: Nimmt aktuelles Waypoint und gibt
        eines der möglichen "next()" zurück.
        """
        if current_wp is None:
            return None
        next_wps = current_wp.next(2.0)  # 2 Meter weiter
        if len(next_wps) == 0:
            return None
        return random.choice(next_wps)

    def _waypoint_ahead_of_vehicle(self, waypoint):
        """
        Prüft grob, ob das gegebene Waypoint noch vor dem Fahrzeug liegt,
        oder ob wir es bereits 'passiert' haben.
        Eine einfache Methode ist, den Vektor (vehicle->waypoint)
        mit der Vorwärtsrichtung des Fahrzeugs zu vergleichen.
        Liegt der Winkel > 90°, ist das Waypoint 'hinter' uns.
        """
        if waypoint is None:
            return True

        vehicle_transform = self.vehicle.get_transform()
        forward_vec = vehicle_transform.get_forward_vector()
        veh_loc = vehicle_transform.location
        wp_loc = waypoint.transform.location

        to_wp = carla.Vector3D(wp_loc.x - veh_loc.x, wp_loc.y - veh_loc.y, 0.0)
        dot = forward_vec.x * to_wp.x + forward_vec.y * to_wp.y

        # Ist das Skalarprodukt < 0 => > 90° => hinter uns
        return dot > 0.0

    def _remove_all_vehicles(self):
        """
        Beispielhafter Reset: Löscht alle NPC-Fahrzeuge (außer unser eigenes),
        kann z.B. aufgerufen werden, wenn das Waypoint hinter uns liegt.
        """
        all_actors = self.world.get_actors()
        vehicles = all_actors.filter("*vehicle*")
        for v in vehicles:
            if v.id != self.vehicle.id:
                v.destroy()

    def _cleanup_actors(self):
        """
        Entfernt Vehicle, Sensoren etc., falls sie noch existieren.
        """
        if self.collision_sensor is not None:
            self.collision_sensor.stop()
            if self.collision_sensor.is_alive:
                self.collision_sensor.destroy()
            self.collision_sensor = None

        if self.camera_sensor is not None:
            self.camera_sensor.stop()
            if self.camera_sensor.is_alive:
                self.camera_sensor.destroy()
            self.camera_sensor = None

        if self.vehicle is not None:
            if self.vehicle.is_alive:
                self.vehicle.destroy()
            self.vehicle = None


#########################
# Beispiel für SB3-Training
#########################

if __name__ == "__main__":
    """
    Kleines Test-Skript, startet die Umgebung und macht ein paar Random-Steps.
    Zur Integration in SB3 etwa:

        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = CarlaGymEnv()
        check_env(env)  # Prüft, ob das Env-Interface korrekt ist

        # Weil wir Dict-Observation haben, MultiInputLstmPolicy wählen:
        model = PPO("MultiInputLstmPolicy", env=DummyVecEnv([lambda: env]), verbose=1)
        model.learn(total_timesteps=100000)
    """

    env = CarlaGymEnv(render_pygame=True)
    obs = env.reset()

    for _ in range(20):
        action = env.action_space.sample()  # random
        obs, reward, done, info = env.step(action)
        env.render()
        print(f"Reward: {reward:.3f}")
        if done:
            obs = env.reset()

    env.close()
