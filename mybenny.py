import torch
import torch.nn as nn
import torch.optim as optim
import umap.umap_ as umap
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting


# --- Autoencoder Definition ---
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

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Autoencoder Training ---
input_size = X.shape[1]
bottleneck_size = 3
autoencoder = Autoencoder(input_size, bottleneck_size).to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
criterion = nn.MSELoss()

X = X.to(device)

num_epochs = 6
batch_size = 256
for epoch in range(num_epochs):
    for i in range(0, X.shape[0], batch_size):
        batch = X[i:min(i + batch_size, X.shape[0])]
        encoded, decoded = autoencoder(batch)
        loss = criterion(decoded, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.9f}')

# --- Extract Bottleneck Features ---
with torch.no_grad():
    bottleneck_features, _ = autoencoder(X)
bottleneck_features = bottleneck_features.cpu().numpy()

# --- UMAP on Bottleneck (3D) ---
umap_obj_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=50, min_dist=0.1)
X_embedded_3d = umap_obj_3d.fit_transform(bottleneck_features)

# --- UMAP on Bottleneck (2D) ---
umap_obj_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=50, min_dist=0.1)
X_embedded_2d = umap_obj_2d.fit_transform(bottleneck_features)


# --- Colormapping by Original Data Index ---
colors = np.arange(X.shape[0])

# --- 2D and 3D Plotting (Simultaneous) ---

# Create figure and subplots
fig = plt.figure(figsize=(16, 8))  # Larger figure for two subplots
ax1 = fig.add_subplot(121)  # 1 row, 2 columns, first subplot (2D)
ax2 = fig.add_subplot(122, projection='3d')  # 1 row, 2 columns, second subplot (3D)


# 2D Plot
ax1.scatter(X_embedded_2d[:, 0], X_embedded_2d[:, 1], c=colors, cmap='winter', s=10)
ax1.set_title("Autoencoder + UMAP (2D)")
cbar1 = fig.colorbar(plt.cm.ScalarMappable(cmap='winter'), ax=ax1) # Create colorbar from ScalarMappable
cbar1.set_label('Data Index')



# 3D Plot
scatter = ax2.scatter(X_embedded_3d[:, 0], X_embedded_3d[:, 1], X_embedded_3d[:, 2], c=colors, cmap='winter', s=10)
ax2.set_title("Autoencoder + UMAP (3D)")
cbar2 = fig.colorbar(scatter, ax=ax2)
cbar2.set_label('Data Index')

# Adjust layout and display
plt.tight_layout()  # Prevent overlapping titles/labels
plt.savefig("combined_plot.eps")
plt.show()

print("Complete")
