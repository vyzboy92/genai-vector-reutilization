# Train a VAE to reduce 1024 dimension vector to 512

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the VAE model
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

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Linear(768, input_dim),
            nn.Sigmoid()  # Assuming input values are normalized to [0, 1]
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# Define the loss function
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence

# Hyperparameters
input_dim = 1024
latent_dim = 512
batch_size = 64
learning_rate = 1e-3
epochs = 20

# Create synthetic data for demonstration
torch.manual_seed(42)
data = torch.randn(1000, input_dim)  # Example dataset with 1000 samples
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer
vae = VariationalAutoencoder(input_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Training loop
vae.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        x = batch[0]

        optimizer.zero_grad()
        recon_x, mu, logvar = vae(x)
        loss = vae_loss(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# Save the trained model
torch.save(vae.state_dict(), "vae_1024_to_512.pth")
