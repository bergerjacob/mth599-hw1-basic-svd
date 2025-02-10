import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def volume_ball(d, R):
    return (np.pi**(d/2) / gamma(d/2 + 1)) * (R**d)

# Radii to plot
radii = [1, 3/2, 2]

# Dimensions
dimensions = np.arange(1, 51)

# Calculate volumes for each radius and dimension
volumes = []
for R in radii:
    volumes.append([volume_ball(d, R) for d in dimensions])

# Create the plot
plt.figure(figsize=(10, 6))

colors = ['blue', 'green', 'red']  # Different colors for each radius
for i, R in enumerate(radii):
    plt.plot(dimensions, volumes[i], label=f'R = {R}', color=colors[i])

plt.xlabel('Dimension (d)')
plt.ylabel('Volume (Vol($B^d(R)$))')
plt.title('Volume of d-dimensional Ball vs. Dimension (Log Scale)')
plt.yscale('log')  # Use a logarithmic scale for the y-axis
plt.grid(True, which="both", ls="--", alpha=0.5) #add grid lines
plt.legend()
plt.savefig('volball.eps', format='eps')
plt.show()
