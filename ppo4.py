import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np

# -----------------------------------------------
# Kaiming- (He-) Initialisierung
# -----------------------------------------------
def kaiming_init(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# -----------------------------------------------
# CNN+LSTM Actor für kontinuierlichen Aktionsraum
# -----------------------------------------------
class CNNLSTMActor(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,         # Bildkanäle, z.B. 3 für RGB
        action_dim: int = 2,          # Dim. des kontinuierlichen Aktionsraums (z.B. [Lenkung, Gas])
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        dropout_prob: float = 0.1,
        dummy_img_height: int = 64,
        dummy_img_width: int = 64,     # Zur Bestimmung der CNN-Feature-Dim
        num_lstm_layers: int = 1      # Anzahl LSTM-Schichten
    ):
        super().__init__()
        # CNN-Feature-Extraktion
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )

        # Feature-Dim automatisch bestimmen
        self.feature_dim = self._get_feature_dim(in_channels, dummy_img_height, dummy_img_width)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # FC-Schicht
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # Actor-Outputs: mu + log_std
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_param = nn.Parameter(torch.zeros(action_dim))

        # Init
        kaiming_init(self)

    def _get_feature_dim(self, in_channels, dummy_img_height, dummy_img_width):
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, dummy_img_height, dummy_img_width)
            out = self.cnn(dummy_input)
            return out.shape[1]

    def forward(self, x, lstm_state=None):
        """
        x: (B, T, C, H, W)  => B = n_envs (oder Batchgröße), T = Zeitschritte
        lstm_state: (h, c) oder None
        Gibt (mu, log_std, (h, c)) zurück.
        """
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B*T, C, H, W)
        feats = self.cnn(x_reshaped)             # => [B*T, feature_dim]
        feats_lstm_in = feats.view(B, T, -1)     # => [B, T, feature_dim]

        if lstm_state is None:
            lstm_out, (h, c) = self.lstm(feats_lstm_in)
        else:
            lstm_out, (h, c) = self.lstm(feats_lstm_in, lstm_state)

        fc_out = self.fc(lstm_out)               # => [B, T, hidden_dim]

        mu = self.mu_head(fc_out)                # => [B, T, action_dim]
        log_std = self.log_std_param.expand_as(mu)
        return mu, log_std, (h, c)

# -----------------------------------------------
# CNN+LSTM Critic (komplett getrennt)
# -----------------------------------------------
class CNNLSTMCritic(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        dropout_prob: float = 0.1,
        dummy_img_height: int = 64,
        dummy_img_width: int = 64,
        num_lstm_layers: int = 1
    ):
        super().__init__()
        # CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )

        self.feature_dim = self._get_feature_dim(in_channels, dummy_img_height, dummy_img_width)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # FC => 1 Wert (Value)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1)
        )

        kaiming_init(self)

    def _get_feature_dim(self, in_channels, dummy_img_height, dummy_img_width):
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, dummy_img_height, dummy_img_width)
            out = self.cnn(dummy_input)
            return out.shape[1]

    def forward(self, x, lstm_state=None):
        """
        x: (B, T, C, H, W)
        Gibt (values, (h, c)) zurück => values in [B, T]
        """
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B*T, C, H, W)
        feats = self.cnn(x_reshaped)             # => [B*T, feature_dim]
        feats_lstm_in = feats.view(B, T, -1)

        if lstm_state is None:
            lstm_out, (h, c) = self.lstm(feats_lstm_in)
        else:
            lstm_out, (h, c) = self.lstm(feats_lstm_in, lstm_state)

        values = self.fc(lstm_out).squeeze(-1)   # => [B, T]
        return values, (h, c)

# -------------------------------------------------
# PPO-Class mit Unterstützung für vectorized envs
# -------------------------------------------------
class PPO:
    """
    Beispiel-Wrapper für CNN+LSTM Actor & Critic.
    Enthält hier nur Basis-Funktionalität, keinen vollständigen Trainingsloop!
    """
    def __init__(
        self,
        in_channels: int = 3,
        dummy_img_height: int = 64,
        dummy_img_width: int = 64,
        action_dim: int = 2,
        init_learning_rate: float = 3e-4,
        lr_konstant: float = 1.0,
        n_maxsteps: int = 100_000,
        roullout: int = 2048,
        n_epochs: int = 10,
        n_envs: int = 1,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Union[None, float] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        device: str = "cpu",
        num_lstm_layers: int = 1
    ):
        self.device = torch.device(device)

        # Speicherung der Hyperparameter
        self.init_learning_rate = init_learning_rate
        self.learning_rate = init_learning_rate
        self.lr_konstant = lr_konstant
        self.n_maxsteps = n_maxsteps
        self.rollout = roullout
        self.n_epochs = n_epochs
        self.n_envs = n_envs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.num_lstm_layers = num_lstm_layers

        # Actor & Critic
        self.actor = CNNLSTMActor(
            in_channels=in_channels,
            action_dim=action_dim,
            num_lstm_layers=num_lstm_layers,
            dummy_img_height=dummy_img_height,
            dummy_img_width=dummy_img_width
        ).to(self.device)

        self.critic = CNNLSTMCritic(
            in_channels=in_channels,
            num_lstm_layers=num_lstm_layers,
            dummy_img_height=dummy_img_height,
            dummy_img_width=dummy_img_width
        ).to(self.device)

        # Optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

    def update_learning_rate(self, epoch):
        """
        Exponentieller LR-Decay
        """
        self.learning_rate = self.init_learning_rate * np.exp(-self.lr_konstant * epoch)
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = self.learning_rate
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def get_action_and_value(self, obs, actor_lstm_state=None, critic_lstm_state=None):
        """
        Rudimentärer Forward-Pass durch Actor & Critic
        obs: (B, T, C, H, W) => B = n_envs
        """
        obs = obs.to(self.device)
        mu, log_std, new_actor_state = self.actor(obs, actor_lstm_state)
        values, new_critic_state = self.critic(obs, critic_lstm_state)
        return mu, log_std, values, new_actor_state, new_critic_state

    def train_on_batch(self, rollouts):
        """
        Platzhalter-Funktion: Hier würdest du
        - die Vorteile (GAE) berechnen
        - ratio = exp(new_logp - old_logp)
        - ppo clip obj.
        - value loss
        - etc.
        """
        pass

    def collect_rollouts(self, vec_env):
        """
        Beispielhafter Pseudocode zum Sammeln von Rollouts aus n_envs parallelen Umgebungen.
        Hier werden LSTM-States verwaltet und pro Schritt die Aktionen ausgeführt.
        Du kannst das für deine Zwecke anpassen (Buffer, GAE usw.).
        """
        # LSTM-Zustände initialisieren (h, c)
        h_actor = torch.zeros(self.num_lstm_layers, self.n_envs, self.actor.lstm.hidden_size, device=self.device)
        c_actor = torch.zeros(self.num_lstm_layers, self.n_envs, self.actor.lstm.hidden_size, device=self.device)
        h_critic = torch.zeros(self.num_lstm_layers, self.n_envs, self.critic.lstm.hidden_size, device=self.device)
        c_critic = torch.zeros(self.num_lstm_layers, self.n_envs, self.critic.lstm.hidden_size, device=self.device)

        # Umgebungen zurücksetzen
        obs = vec_env.reset()  # => shape (n_envs, C, H, W)
        rollouts = []

        for step in range(self.rollout):
            # Die Beobachtung in Batch-Form bringen => (n_envs, 1, C, H, W)
            obs_torch = torch.tensor(obs, dtype=torch.float32).unsqueeze(1)  # (n_envs, 1, C, H, W)

            # Aktion + Value schätzen
            mu, log_std, values, (h_actor, c_actor), (h_critic, c_critic) = self.get_action_and_value(
                obs_torch, (h_actor, c_actor), (h_critic, c_critic)
            )

            # Aus mu + log_std eine Aktion samplen (hier einfach Normal-Verteilung)
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()[:, 0, :]  # => shape (n_envs, action_dim)

            # Schritt in der Umgebung ausführen
            action_np = action.cpu().numpy()
            next_obs, rewards, dones, infos = vec_env.step(action_np)

            # Alles für Training zwischenspeichern (hier nur Pseudocode)
            rollouts.append({
                "obs": obs,
                "action": action_np,
                "reward": rewards,
                "done": dones,
                "value": values[:, 0].detach().cpu().numpy(),  # shape (n_envs,)
                # LSTM-States werden in Praxis evtl. separat verwaltet/gespeichert
            })

            # Falls eine Umgebung done ist, resetten
            for i, done in enumerate(dones):
                if done:
                    # LSTM-States für diese eine Environment i zurücksetzen
                    h_actor[:, i, :] = 0
                    c_actor[:, i, :] = 0
                    h_critic[:, i, :] = 0
                    c_critic[:, i, :] = 0

            obs = next_obs

        # Rückgabe: Diese rollouts kannst du dann an `train_on_batch` übergeben
        return rollouts
