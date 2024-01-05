import matplotlib.pyplot as plt
from sbi.analysis import pairplot
import torch

def trajectory_plot(trajectory, ax):
    trajectory = trajectory.numpy()
    trajectory_x = trajectory[:,0]
    trajectory_y = trajectory[:,1]
    ax.plot(trajectory_x, trajectory_y)
    ax.set_aspect('equal')

def multiple_trajectory_plot(trajectories):
    fig, ax = plt.subplots(figsize=(8, 8))
    for trajectory in trajectories:
        trajectory_plot(trajectory, ax)

def density_plot(samples):
    samples_switched = torch.stack([samples[:,1],samples[:,0]],dim=1)
    pairplot(samples_switched)


