# Import necessary libraries
import numpy as np
from mplanner.core.environment import DMove, Grid
from mplanner.algorithms.astar import AStar
from mplanner.core.utils import mark_obstacles, calculate_grid_size, to_grid_coords

# Configuration Parameters
resolution = 0.8  # Grid cell size in meters
environment_bounds = {
    "min": (-5, 0, -5),  # Minimum bounds of the environment in Unity coordinates
    "max": (5, 5, 5),    # Maximum bounds of the environment in Unity coordinates
}
origin = environment_bounds["min"]  # Origin set as the minimum bounds

# Calculate the grid size automatically based on the environment bounds and resolution
grid_size = calculate_grid_size(environment_bounds, resolution)

# Initialize the grid matrix with 1s (walkable areas)
matrix = np.ones(grid_size, dtype=np.int8)

# Define obstacles in the environment
# Each obstacle is defined by its center position and dimensions in Unity coordinates
obstacles = [
    ((1.63, 1.82, 1.59), (1.00, 1.00, 1.00)),  # Cube_1
    ((-1.31, 1.85, -1.82), (1.00, 1.00, 1.00)),  # Cube_2
]

# Mark the obstacles in the grid
for position, dimensions in obstacles:
    mark_obstacles(matrix, position, dimensions, resolution, origin)

# Create the grid object
grid = Grid(matrix=matrix)

# Define the start and end positions in Unity coordinates
start_position = (2.16, 2.24, -2.88)  # Start position
end_position = (-1.05, 1.96, 3.09)    # End position

# Convert the start and end positions to grid coordinates
start_coords = to_grid_coords(start_position, resolution, origin)
end_coords = to_grid_coords(end_position, resolution, origin)

# Create start and end nodes in the grid
start = grid.node(*start_coords)
end = grid.node(*end_coords)

# Initialize the A* path planner
planner = AStar(diagonal_movement=DMove.NEVER)

# Run the A* algorithm to compute the path
path, runs = planner.get_planned_path(start, end, grid)

# Convert the path back to Unity world coordinates
waypoints = [
    ((origin[0] + p.x * resolution).__round__(2), (origin[1] + p.y * resolution).__round__(2), (origin[2] + p.z * resolution).__round__(2))
    for p in path
]

print("Operations:", runs, "Path Length:", len(path))
print("List of waypoints")
for p in waypoints:
    print(p)