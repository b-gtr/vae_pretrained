"""
File: use_pretrained_vae.py

Kurzes Beispiel, wie man ein bereits trainiertes VAE mit
(var_autoencoder.pth, var_encoder_model.pth, decoder_model.pth)
lädt und daraus nur den Encoder-Latent-Vektor nutzt.

Dieses Script können Sie direkt so abändern/anpassen und
in Ihrem RL-Code weiterverwenden.
"""

import torch
import numpy as np
from vae import VariationalAutoencoder  # Ihr "Haupt-Wrapper", der encoder und decoder enthält

# Geben Sie hier Ihren GPU/CPU-Device an
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------------
# 1) VAE-Modell laden
# ------------------------------------------------------------------------------------
LATENT_SPACE = 95
vae_model = VariationalAutoencoder(latent_dims=LATENT_SPACE).to(device)

# Laden Sie bevorzugt var_autoencoder.pth, das sollte alle Gewichte enthalten
vae_model_path = "autoencoder/model/var_autoencoder.pth"  
vae_model.load_state_dict(torch.load(vae_model_path, map_location=device))

# Alternativ könnte man einzeln laden:
# vae_model.encoder.load()
# vae_model.decoder.load()
# ... wenn man z. B. var_encoder_model.pth und decoder_model.pth separat hat.

vae_model.eval()
# Encoder-Parameter einfrieren, sodass im RL keine Gewichte geändert werden
for param in vae_model.parameters():
    param.requires_grad = False

print("VAE (Encoder+Decoder) erfolgreich geladen.")

# ------------------------------------------------------------------------------------
# 2) Beispiel: Latent-Vektor extrahieren
# ------------------------------------------------------------------------------------
# Angenommen, wir haben irgendein RGB-Image (z. B. aus CARLA-Kamera),
# shape [3, 80, 160], Wertebereich [0..1].
# Hier nutzen wir ein Fake-Array nur als Demo:
dummy_image = np.random.rand(3, 80, 160).astype(np.float32)

# In Torch konvertieren
dummy_tensor = torch.from_numpy(dummy_image).unsqueeze(0).to(device)
# => shape [1, 3, 80, 160]

with torch.no_grad():
    # Vorwärtsdurchlauf des Encoders. Standard: z = mu + sigma*eps
    # z = vae_model.encoder(dummy_tensor)
    
    # (Falls Sie deterministisch vorgehen wollen, 
    #  können Sie "encode_deterministic" implementieren, siehe Kommentar unten.)
    z = vae_model.encoder(dummy_tensor)

print(f"Latent Vector z Shape: {z.shape}")  # => [1, 95]

# CPU-Konvertierung, um z z. B. im Numpy-Format zu verarbeiten
z_np = z.squeeze(0).cpu().numpy()  # => shape (95,)
print("Latent Vektor:", z_np)

# ------------------------------------------------------------------------------------
# 3) (Optional) Rekonstruktion (Decoder)
# ------------------------------------------------------------------------------------
# Nur zum Debuggen/Anschauen. 
# Decoder nimmt z als Eingabe und gibt ein rekonstruiertes Bild (3-Kanal) aus.
with torch.no_grad():
    x_hat = vae_model.decoder(z)  # shape => [1, 3, 80+..., 160+...] (abh. von ConvTranspose)
print("Reconstruction shape:", x_hat.shape)

# ------------------------------------------------------------------------------------
# HINWEIS für deterministisches Encoding (falls gewünscht):
# ------------------------------------------------------------------------------------
# Sie könnten in encoder.py Folgendes ergänzen:
#
#   def encode_deterministic(self, x):
#       x = x.to(device)
#       x = self.encoder_layer1(x)
#       x = self.encoder_layer2(x)
#       x = self.encoder_layer3(x)
#       x = self.encoder_layer4(x)
#       x = torch.flatten(x, start_dim=1)
#       x = self.linear(x)
#       mu = self.mu(x)
#       # sigma = torch.exp(self.sigma(x))  # ignorieren
#       # kein sampling
#       return mu
#
# Dann aufrufen: z_det = vae_model.encoder.encode_deterministic(dummy_tensor)
#
# So erhalten Sie *immer* das gleiche z ohne Zufallsstörung.
#
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Demo abgeschlossen.")
