import torch
import torch.nn as nn

# Define the VAE model class (same as above)
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(512, latent_dim)  # Mean vector of the latent space
        self.fc_logvar = nn.Linear(512, latent_dim)  # Log variance of the latent space

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)  # logvar can be ignored for dimensionality reduction
        return mu, logvar

# Initialize the model
input_dim = 1024
latent_dim = 512
vae = VariationalAutoencoder(input_dim, latent_dim)

# Load the saved weights
vae.load_state_dict(torch.load("vae_1024_to_512.pth"))
vae.eval()  # Set the model to evaluation mode

# Example input vector for dimensionality reduction
input_vector = torch.randn(1, input_dim)  # Replace with your actual 1024-dimensional vector

# Reduce the dimension
with torch.no_grad():  # No gradients required
    mu, _ = vae.encode(input_vector)  # Extract the mean (mu) as the reduced representation
    reduced_vector = mu

print("Original vector shape:", input_vector.shape)
print("Reduced vector shape:", reduced_vector.shape)
