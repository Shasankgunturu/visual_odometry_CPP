# import numpy as np
# import matplotlib.pyplot as plt
# def plot_poses(poses):
#     fig, ax = plt.subplots()
#     ax.scatter(poses[:, 0], poses[:, 1], c='r', marker='o')
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     plt.show()
# filename = "/home/shasankgunturu/personal/ComputerVisionBasics/src/output/poses1.txt"
# poses = np.loadtxt(filename) 
# plot_poses(poses)


import numpy as np
import matplotlib.pyplot as plt

def load_poses_from_txt(filename, num_rows=500):
    """Load transformation matrices from a text file and extract the (x, y, z) translation part."""
    data = np.loadtxt(filename, max_rows=num_rows)
    translations = data.reshape(-1, 3, 4)[:, :, 3]
    return translations

def load_poses_from_poses_file(filename):
    """Load (x, y, z) poses directly from poses1.txt."""
    return np.loadtxt(filename)

def plot_poses(poses, label):
    """Plot the given poses."""
    fig, ax = plt.subplots()
    ax.scatter(poses[:, 0], poses[:, 1], c='r', marker='o', label=label, s=1)  # 's=10' sets marker size to small
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.legend()
    plt.show()

def compare_poses(poses1, poses2):
    """Compare two sets of poses by plotting them together."""
    fig, ax = plt.subplots()
    ax.scatter(poses1[:, 0], poses1[:, 1], c='r', marker='o', label='Original Poses', s=5)  # Set size to small
    ax.scatter(poses2[:, 0], poses2[:, 2], c='b', marker='x', label='Extracted Poses', s=5)  # Set size to small
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.legend()
    plt.show()

# Load the extracted poses from xx.txt
filename_xx = "/home/shasank-gunturu/Downloads/data_odometry_poses/dataset/poses/00.txt"  # Replace with the actual path to xx.txt
extracted_poses = load_poses_from_txt(filename_xx)

# Load the poses from poses1.txt
filename_poses1 = "/home/shasank-gunturu/personal/visual_odometry_CPP/output/poses1.txt"  # Replace with the actual path to poses1.txt
original_poses = load_poses_from_poses_file(filename_poses1)

# Compare and plot the poses
compare_poses(original_poses, extracted_poses)
