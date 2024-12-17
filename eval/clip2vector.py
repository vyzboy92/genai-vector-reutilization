import torch
import torch.nn as nn
from llm2vector import VariationalAutoencoder

# Define the modified VAE for feature extraction
class ModifiedVAE(nn.Module):
    def __init__(self, original_vae, new_input_dim):
        super(ModifiedVAE, self).__init__()

        # Modify encoder's first layer
        self.encoder = nn.Sequential(
            nn.Linear(new_input_dim, 768),  # New input layer
            *list(original_vae.encoder.children())[1:]  # Reuse the rest of the encoder
        )

        self.fc_mu = original_vae.fc_mu

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        return mu  # Latent features

# New input dimension
new_input_dim = 512  # Example: reducing input size to 512
latent_dim = 512

# Load the original VAE
original_vae = VariationalAutoencoder(input_dim=1024, latent_dim=latent_dim)
original_vae.load_state_dict(torch.load("vae_1024_to_512.pth"))
original_vae.eval()

# Create the modified VAE for feature extraction
feature_extractor = ModifiedVAE(original_vae, new_input_dim)
feature_extractor.eval()

# Example input with new dimension
new_input_vector = torch.randn(1, new_input_dim)  # Replace with your actual data

# Extract features using the encoder
with torch.no_grad():
    extracted_features = feature_extractor.encode(new_input_vector)

print("Original input vector shape:", new_input_vector.shape)
print("Extracted features shape:", extracted_features.shape)
