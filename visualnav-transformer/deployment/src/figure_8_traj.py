import numpy as np

def generate_figure_8_trajectory(num_points, scale=1.0, altitude=-10):
    """
    Generate a figure-8 trajectory.
    
    :param num_points: Number of points in the trajectory.
    :param scale: Scale of the figure-8.
    :param altitude: Altitude of the trajectory.
    :return: List of (x, y, z) tuples representing the trajectory.
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    x = scale * np.sin(t)
    y = scale * np.sin(t) * np.cos(t)
    z = np.full_like(x, altitude)  # Constant altitude

    trajectory = list(zip(x, y, z))
    return trajectory

def save_trajectory(trajectory, filename):
    """
    Save the trajectory to a file.
    
    :param trajectory: List of (x, y, z) tuples representing the trajectory.
    :param filename: Name of the file to save the trajectory.
    """
    with open(filename, 'w') as f:
        for point in trajectory:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

# Generate and save the trajectory
num_points = 100  # Adjust the number of points as needed
trajectory = generate_figure_8_trajectory(num_points, scale=10, altitude=-50)  # Scale and altitude can be adjusted
save_trajectory(trajectory, "figure_8_trajectory.txt")
