import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/training.log"),
        logging.StreamHandler()
    ]
)

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

# Data preparation
def prepare_data(input_dim, batch_size, filename):
    try:
        # Load the saved tensor (for testing or reuse)
        logging.info(f"Loading embeddings from {filename}.")
        embeddings = torch.load(filename)

        if embeddings.shape[1] != input_dim:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match input_dim {input_dim}.")

        dataset = TensorDataset(embeddings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    except FileNotFoundError:
        logging.error(f"File not found: {filename}")
        raise
    except Exception as e:
        logging.error(f"Error during data preparation: {e}")
        raise

# Model and optimizer initialization
def initialize_model(input_dim, latent_dim, learning_rate):
    try:
        logging.info("Initializing Variational Autoencoder and optimizer.")
        vae = VariationalAutoencoder(input_dim, latent_dim)
        optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
        return vae, optimizer
    except Exception as e:
        logging.error(f"Error during model initialization: {e}")
        raise

# Training loop
def train_model(vae, optimizer, dataloader, epochs):
    try:
        logging.info("Starting training loop.")
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

            logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

# Save the trained model
def save_model(vae, model_path):
    try:
        logging.info(f"Saving trained model to {model_path}.")
        torch.save(vae.state_dict(), model_path)
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.error(f"Error during model saving: {e}")
        raise

if __name__ == "__main__":
    # Hyperparameters
    input_dim = 1024
    latent_dim = 512
    batch_size = 64
    learning_rate = 1e-3
    epochs = 20
    model_path = "vae_1024_to_512.pth"
    embedding_file = "./saved_weights/sentence_embeddings.pt"  # Path to the text file containing sentences

    try:
        dataloader = prepare_data(input_dim, batch_size, embedding_file)
        vae, optimizer = initialize_model(input_dim, latent_dim, learning_rate)
        train_model(vae, optimizer, dataloader, epochs)
        save_model(vae, model_path)
    except Exception as e:
        logging.critical(f"Training pipeline failed: {e}")
