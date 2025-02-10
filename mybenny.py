import torch
import torch.nn as nn
import torch.optim as optim
import umap
import matplotlib.pyplot as plt
import numpy as np

# --- Autoencoder Definition (This is probably in some library but whatever...) ---
class Autoencoder(nn.Module):
    def __init__(self, input_size, bottleneck_size=3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, bottleneck_size)  # 3D bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()  # Output between 0 and 1 (pixel values)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# --- Data Loading and Preprocessing ---
X = torch.load('benny_v2.pt')
sz = X.shape[1:]
X = X.reshape(X.shape[0], -1)
X = X.float() / 255.

# --- Device Configuration (GPU will be a lot faster if available) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Autoencoder Training ---
input_size = X.shape[1]
bottleneck_size = 3
autoencoder = Autoencoder(input_size, bottleneck_size).to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
criterion = nn.MSELoss()

X = X.to(device)

num_epochs = 30
batch_size = 256
for epoch in range(num_epochs):
    for i in range(0, X.shape[0], batch_size):
        batch = X[i:min(i + batch_size, X.shape[0])]
        encoded, decoded = autoencoder(batch)
        loss = criterion(decoded, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# --- Extract Bottleneck Features ---
with torch.no_grad():
    bottleneck_features, _ = autoencoder(X)
bottleneck_features = bottleneck_features.cpu().numpy()

# --- UMAP on Bottleneck ---
umap_obj = umap.UMAP(n_components=2, random_state=42, n_neighbors=50, min_dist=0.1)
X_embedded = umap_obj.fit_transform(bottleneck_features)

# --- Colormapping by Original Data Index ---
colors = np.arange(X.shape[0])

# --- Plotting ---
plt.figure(figsize=(8, 6))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors, cmap='winter', s=10)
plt.colorbar(label='Data Index')
plt.title("Autoencoder + UMAP (Colored by Data Index)")
plt.savefig("after.eps")
plt.show()

# --- PCA for comparison---
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.cpu().detach().numpy())

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=10)
plt.title("PCA")
plt.savefig("before.eps")
plt.show()

print("Complete")
