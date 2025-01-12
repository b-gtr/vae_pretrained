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
        in_channels: int = 3,         # z.B. 3 für RGB
        action_dim: int = 2,          # Dim. des kontinuierlichen Aktionsraums
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        dropout_prob: float = 0.1,
        dummy_img_height: int = 64,
        dummy_img_width: int = 64,    
        num_lstm_layers: int = 1      
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
        x: (B, T, C, H, W)
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
        Gibt (values, (h, c)) zurück => values in [B, T].
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
# PPO-Class
# -------------------------------------------------
class PPO:
    """
    PPO-Wrapper für CNN+LSTM Actor & Critic.
    - mit GAE
    - PPO-Clipping
    - Value-Clipping (optional)
    - Entropie-Bonus
    - Mini-Batch + n_epochs
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
        rollout_steps: int = 2048,
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
        self.rollout_steps = rollout_steps
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
        Forward-Pass durch Actor & Critic
        obs: (B, T, C, H, W) => B = n_envs
        """
        obs = obs.to(self.device)
        mu, log_std, new_actor_state = self.actor(obs, actor_lstm_state)
        values, new_critic_state = self.critic(obs, critic_lstm_state)
        return mu, log_std, values, new_actor_state, new_critic_state

    # -------------------------------------------------
    # GAE-Berechnung
    # -------------------------------------------------
    def compute_gae(self, rollouts, last_values):
        """
        Berechnet GAE und Vtarget rückwärts über die Rollouts.

        rollouts: Liste von Dicts (pro Environment-Schritt):
            {
              "obs": (C, H, W),
              "action": ...,
              "reward": r_t,
              "done": bool,
              "value": V_t,
              "log_prob": \log \pi_{\text{old}}(a_t|s_t)
            }

        last_values: Critic-Wert für den Zustand NACH dem letzten Step
        """
        T = len(rollouts)
        advantages = np.zeros((T,), dtype=np.float32)

        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_values
                next_done = rollouts[t]["done"]
            else:
                next_value = rollouts[t+1]["value"]
                next_done = rollouts[t+1]["done"]

            reward = rollouts[t]["reward"]
            value = rollouts[t]["value"]

            # delta = r_t + gamma * V_{t+1} * (1-done) - V_t
            delta = reward + self.gamma * (1 - float(next_done)) * next_value - value
            # gae = delta + gamma*lambda * (1-done) * gae
            gae = delta + self.gamma * self.gae_lambda * (1 - float(next_done)) * gae
            advantages[t] = gae

        for t in range(T):
            rollouts[t]["advantage"] = advantages[t]
            rollouts[t]["return"] = rollouts[t]["value"] + advantages[t]

        return rollouts

    # -------------------------------------------------
    # PPO-Trainingsschritt
    # -------------------------------------------------
    def train_on_batch(self, rollouts):
        """
        - 1) Value für den "nächsten" Zustand schätzen
        - 2) GAE berechnen
        - 3) Mini-Batch + n_epochs
        - 4) PPO-Clipping-Loss, optional Value-Clipping, Entropie
        """
        if len(rollouts) == 0:
            return {}

        # 1) Kritiker-Wert für letzten Zustand
        last_obs = rollouts[-1]["obs"]
        obs_torch = torch.tensor(last_obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Je nach Net-Layout ggf. permute
        obs_torch = torch.permute(obs_torch, (0, 1, 4, 2, 3))  
        with torch.no_grad():
            value_final, _ = self.critic(obs_torch)
        last_value = value_final[0, 0].cpu().numpy()
        if rollouts[-1]["done"]:
            last_value = 0.0

        # 2) GAE & Returns
        rollouts = self.compute_gae(rollouts, last_value)

        # -----------------------------------------
        # Daten in Tensoren packen
        # -----------------------------------------
        obs_list     = []
        actions_list = []
        logp_list    = []
        adv_list     = []
        returns_list = []
        values_list  = []

        for r in rollouts:
            obs_list.append(r["obs"])
            actions_list.append(r["action"])
            logp_list.append(r["log_prob"])
            adv_list.append(r["advantage"])
            returns_list.append(r["return"])
            values_list.append(r["value"])

        obs_tensor      = torch.tensor(obs_list,      dtype=torch.float32, device=self.device)  
        actions_tensor  = torch.tensor(actions_list,  dtype=torch.float32, device=self.device)  
        old_logp_tensor = torch.tensor(logp_list,     dtype=torch.float32, device=self.device)  
        advantages_tensor = torch.tensor(adv_list,    dtype=torch.float32, device=self.device)  
        returns_tensor    = torch.tensor(returns_list,dtype=torch.float32, device=self.device)  
        old_values_tensor = torch.tensor(values_list, dtype=torch.float32, device=self.device)  

        # Für LSTM-Forward (T,1,C,H,W). Hier T = #Rollouts. 
        # (=> Eigentlich bräuchte man zeitliche Sequenzen pro Env)
        obs_tensor = obs_tensor.unsqueeze(1)

        # Normalisieren der Vorteile (oft verwendet für Stabilität):
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # ------------------------------------------------
        # Mehrfache Epochen + Mini-Batch-Verarbeitung
        # ------------------------------------------------
        T = len(rollouts)
        inds = np.arange(T)

        # Sammeln für Logging
        final_actor_loss = 0.0
        final_value_loss = 0.0
        final_entropy    = 0.0

        for epoch_i in range(self.n_epochs):
            np.random.shuffle(inds)
            start_idx = 0
            while start_idx < T:
                end_idx = start_idx + self.batch_size
                batch_idx = inds[start_idx:end_idx]

                # --- Mini-Batch aus Rollouts ---
                obs_b       = obs_tensor[batch_idx]     # => shape (B,1,C,H,W)
                act_b       = actions_tensor[batch_idx]  # => shape (B, action_dim)
                old_logp_b  = old_logp_tensor[batch_idx] # => shape (B,)
                adv_b       = advantages_tensor[batch_idx]
                ret_b       = returns_tensor[batch_idx]
                old_val_b   = old_values_tensor[batch_idx]

                # --- Forward Pass ---
                mu_b, log_std_b, _  = self.actor(obs_b)
                val_b, _            = self.critic(obs_b)

                # Flatten T=1
                mu_b   = mu_b[:, 0, :]   # => (B, action_dim)
                val_b  = val_b[:, 0]     # => (B,)

                std_b  = log_std_b.exp() # => (B, 1, action_dim) => (B,1,action_dim)
                dist_b = torch.distributions.Normal(mu_b, std_b)

                # Neue log_probs
                new_logp_b = dist_b.log_prob(act_b).sum(dim=-1)  # => (B,)

                # Ratio
                ratio_b = torch.exp(new_logp_b - old_logp_b)

                # Policy-Clipping
                clipped_ratio_b = torch.clamp(ratio_b, 1.0 - self.clip_range, 1.0 + self.clip_range)
                # Zwei Möglichkeiten
                loss_unclipped = ratio_b        * adv_b
                loss_clipped   = clipped_ratio_b * adv_b

                policy_loss = -torch.mean(torch.min(loss_unclipped, loss_clipped))

                # Entropie-Bonus
                entropy_b = dist_b.entropy().sum(dim=-1)  # => (B,)
                entropy_loss = -self.ent_coef * entropy_b.mean()

                # => Finaler Actor-Loss
                actor_loss = policy_loss + entropy_loss

                # Value-Clipping (optional)
                if self.clip_range_vf is not None:
                    # Clip predicted value
                    val_clipped = old_val_b + torch.clamp(val_b - old_val_b,
                                                         -self.clip_range_vf, self.clip_range_vf)
                    value_unclipped = (val_b - ret_b)**2
                    value_clipped   = (val_clipped - ret_b)**2
                    value_loss = 0.5 * torch.mean(torch.max(value_unclipped, value_clipped))
                else:
                    # Ohne Value-Clipping
                    value_loss = 0.5 * F.mse_loss(val_b, ret_b)

                total_loss = actor_loss + self.vf_coef * value_loss

                # --- Backward & Update ---
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                final_actor_loss = actor_loss.item()
                final_value_loss = value_loss.item()
                final_entropy    = entropy_b.mean().item()

                start_idx = end_idx

        info = {
            "final_actor_loss": final_actor_loss,
            "final_value_loss": final_value_loss,
            "final_entropy": final_entropy
        }
        return info

    # -------------------------------------------------
    # Rollout-Sammlung
    # -------------------------------------------------
    def collect_rollouts(self, vec_env):
        """
        Beispielhafter Pseudocode zum Sammeln von Rollouts aus n_envs parallelen Umgebungen.
        LSTM-States werden verwaltet.
        """
        h_actor = torch.zeros(self.num_lstm_layers, self.n_envs, self.actor.lstm.hidden_size, device=self.device)
        c_actor = torch.zeros(self.num_lstm_layers, self.n_envs, self.actor.lstm.hidden_size, device=self.device)
        h_critic = torch.zeros(self.num_lstm_layers, self.n_envs, self.critic.lstm.hidden_size, device=self.device)
        c_critic = torch.zeros(self.num_lstm_layers, self.n_envs, self.critic.lstm.hidden_size, device=self.device)

        obs = vec_env.reset()  # => shape (n_envs, C, H, W)
        rollouts = []

        for step in range(self.rollout_steps):
            # (n_envs, 1, C, H, W)
            obs_torch = torch.tensor(obs, dtype=torch.float32).unsqueeze(1)
            # Ggf. permute je nach Net
            obs_torch = torch.permute(obs_torch, (0, 1, 4, 2, 3))

            mu, log_std, values, (h_actor, c_actor), (h_critic, c_critic) = self.get_action_and_value(
                obs_torch, (h_actor, c_actor), (h_critic, c_critic)
            )

            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()[:, 0, :]           # => shape (n_envs, action_dim)
            log_prob = dist.log_prob(action).sum(-1) # => shape (n_envs,)

            action_np = action.cpu().numpy()
            next_obs, rewards, dones, infos = vec_env.step(action_np)

            for i in range(self.n_envs):
                rollouts.append({
                    "obs": obs[i],
                    "action": action_np[i],
                    "reward": rewards[i],
                    "done": dones[i],
                    "value": values[i, 0].item(),
                    "log_prob": log_prob[i].item()
                })

            # LSTM-Zustand zurücksetzen pro done
            for i, done in enumerate(dones):
                if done:
                    h_actor[:, i, :] = 0
                    c_actor[:, i, :] = 0
                    h_critic[:, i, :] = 0
                    c_critic[:, i, :] = 0

            obs = next_obs

        return rollouts
