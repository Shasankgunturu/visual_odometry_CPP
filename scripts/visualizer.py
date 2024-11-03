import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt("/home/shasank-gunturu/personal/visual_odometry_CPP/output/poses1.txt")
ground_truth = data[:, 0:2]  # Only x and z
estimated = data[:, 2:4]     # Only x and z

# Compute the positional error at each point (Euclidean distance in 2D)
errors = np.linalg.norm(ground_truth - estimated, axis=1)

# Plot trajectories
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].plot(ground_truth[:, 0], ground_truth[:, 1], label='Ground Truth', color='blue')
ax[0].plot(estimated[:, 0], estimated[:, 1], label='Estimated', color='cyan')
ax[0].set_title('Trajectories (X-Z Plane)')
ax[0].set_xlabel('X')
ax[0].set_ylabel('Z')
ax[0].legend()
ax[0].axis('equal')

# Plot error over time
ax[1].plot(errors, label='Positional Error (X-Z)', color='red')
ax[1].set_title('Positional Error over Time (X-Z Plane)')
ax[1].set_xlabel('Timestep')
ax[1].set_ylabel('Error (Euclidean distance)')
ax[1].legend()

plt.tight_layout()
plt.show()
