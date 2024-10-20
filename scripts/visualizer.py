import numpy as np
import matplotlib.pyplot as plt
def plot_poses(poses):
    fig, ax = plt.subplots()
    ax.scatter(poses[:, 0], poses[:, 1], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.show()
filename = "/home/shasankgunturu/personal/ComputerVisionBasics/src/output/poses1.txt"
poses = np.loadtxt(filename) 
plot_poses(poses)
