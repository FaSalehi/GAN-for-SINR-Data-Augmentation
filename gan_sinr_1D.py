import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scipy.io
from JS_divergence import compute_jsd

loaded_object = scipy.io.loadmat('sinr_dataset')
sinr_map = loaded_object['sinr_map']

# Define the generator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Define the discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Hyperparameters
input_dim = 1
noise_dim = 10    # the dimension of the noise vector
batch_size = 128   # the number of samples to generate
epochs = 3000
lr = 0.0002

# Prepare data
# Convert sinr_map to a list of 2D arrays
sinr_list = [sinr_map[0, i] for i in range(sinr_map.shape[1])]

# Flatten each 60x120 matrix to a 1D array of length 7200
flat_sinr = np.array([arr.flatten() for arr in sinr_list], dtype=np.float32)  # shape (355, 7200)
real_data = torch.tensor(flat_sinr).view(-1, 1)  # shape (355*7200, 1)

# Create GAN
G = Generator(noise_dim, input_dim)
D = Discriminator(input_dim)

criterion = nn.BCELoss()
optimizerD = optim.Adam(D.parameters(), lr=lr)
optimizerG = optim.Adam(G.parameters(), lr=lr)

# Lists to store losses
D_real_losses = []
D_fake_losses = []
G_losses = []

# Training loop
for epoch in range(epochs):
    # Train Discriminator
    optimizerD.zero_grad()
    
    # Real data
    real_labels = torch.ones(batch_size, 1)
    real_sample = real_data[torch.randint(0, len(real_data), (batch_size,))]
    output_real = D(real_sample)
    loss_real = criterion(output_real, real_labels)
    
    # Fake data
    noise = torch.randn(batch_size, noise_dim)
    fake_sample = G(noise)
    fake_labels = torch.zeros(batch_size, 1)
    output_fake = D(fake_sample.detach())
    loss_fake = criterion(output_fake, fake_labels)
    
    # Backpropagation and optimization
    lossD = loss_real + loss_fake
    lossD.backward()
    optimizerD.step()
    
    # Train Generator
    optimizerG.zero_grad()
    
    output_fake = D(fake_sample)
    lossG = criterion(output_fake, real_labels)
    
    lossG.backward()
    optimizerG.step()

    # Record losses
    D_real_losses.append(loss_real.item())
    D_fake_losses.append(loss_fake.item())
    G_losses.append(lossG.item())
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss D: {lossD.item()}, Loss G: {lossG.item()}')

# Plot the loss evolution
plt.figure(figsize=(10, 5))
plt.plot(D_real_losses, label='Discriminator Real Loss')
plt.plot(D_fake_losses, label='Discriminator Fake Loss')
plt.plot(G_losses, label='Generator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Evolution During Training')
plt.legend()
plt.show()

# Generate data and estimate PDF
with torch.no_grad():
    noise = torch.randn(real_data.size(0), noise_dim)
    generated_data = G(noise).numpy().flatten()

# Plot the results
plt.hist(generated_data, bins=50, density=True, alpha=0.5, label='Generated Data')
kde = gaussian_kde(real_data.numpy().flatten())
x = np.linspace(real_data.min(), real_data.max(), 1000)
plt.plot(x, kde(x), label='Real Data KDE')
plt.legend()
plt.show()

# Calculate Jensen-Shannon Divergence
jsd_value = compute_jsd(real_data, generated_data)
print(f"Jensen-Shannon Divergence: {jsd_value:.4f}")
