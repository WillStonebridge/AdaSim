import sys
import json
import csv
import numpy as np
from mplanner.core.environment import DMove, Grid
from mplanner.algorithms.astar import AStar
from mplanner.algorithms.dijkstra import Dijkstra3D
from mplanner.core.utils import mark_obstacles, calculate_grid_size, to_grid_coords

def within_boundary(obj, enviroment_bounds):
    X_within = enviroment_bounds['min'][0] < obj['position']['X'] - obj['dimension']['X'] and \
          obj['position']['X'] + obj['dimension']['X'] < enviroment_bounds['max'][0]
    
    # Y_within = enviroment_bounds['min'][1] < obj['position']['Y'] and \
    #       obj['position']['Y'] < enviroment_bounds['max'][1]
    
    Z_within = enviroment_bounds['min'][2] < obj['position']['Z'] - obj['dimension']['Z'] and \
          obj['position']['Z'] + obj['dimension']['Z'] < enviroment_bounds['max'][2]
    
    return X_within and Z_within #and Y_within
    


if __name__ == "__main__":
    scene_data_path = sys.argv[1]
    waypoint_path = sys.argv[2]

    # Open and load the JSON file
    with open(scene_data_path, 'r') as file:
        scene = json.load(file)

    # load in the start and end positions
    robot = scene['robot']
    goal = scene['goal']
    start_position = (robot["position"]['X'],
                      robot["position"]['Y'],
                      robot["position"]['Z'])
    end_position = (goal["position"]['X'],
                    goal["position"]['Y'],
                    goal["position"]['Z'])

    #TODO: the boundary box is determined in 2d space, its ceiling is arbitarily determined by the goal
    #TODO: make it an actual box
    boundary_box = scene['boundary']
    environment_bounds = {
        "min": (boundary_box['position']['X'] - boundary_box['dimension']['X']/2,
                boundary_box['position']['Y'] - boundary_box['dimension']['Y']/2,
                boundary_box['position']['Z'] - boundary_box['dimension']['Z']/2),
        "max": (boundary_box['position']['X'] + boundary_box['dimension']['X']/2,
                boundary_box['position']['Y'] + boundary_box['dimension']['Y']/2,
                boundary_box['position']['Z'] + boundary_box['dimension']['Z']/2)
    } #40*goal['position']['Y']+goal['dimension']['Y'],
    print(f"ROBOT: pos-{robot['position']}, dim-{robot['dimension']}")
    print(f"env bounds - {environment_bounds}")

    #resolution is determined based on the largest dimension of the robot, this prevents it from hitting corners
    resolution = 0.4 #max(robot['dimension'].values())

    origin = environment_bounds["min"]  # Origin set as the minimum bounds

    # Calculate the grid size automatically based on the environment bounds and resolution
    grid_size = calculate_grid_size(environment_bounds, resolution)
    print(f"grid shape - {grid_size}")

    # Initialize the grid matrix with 1s (walkable areas)
    matrix = np.ones(grid_size, dtype=np.int8)

    # Define obstacles in the environment
    # Each obstacle is defined by its center position and dimensions in Unity coordinates
    for object_name, object_data in scene.items():
        if not object_name in ["robot", "goal", "boundary"]: #These objects names are reserved
            # if object_name == "Building_Stadium":
            #     breakpoint()
            if True: #within_boundary(object_data, environment_bounds):
                object_position = (object_data["position"]['X'],
                                   object_data["position"]['Y'],
                                   object_data["position"]['Z'])
                object_dimension = (object_data["dimension"]['X']+1*robot['dimension']['X'],
                                    object_data["dimension"]['Y'],
                                    object_data["dimension"]['Z']+1*robot['dimension']['Z'],)

                # Mark the obstacles in the grid
                mark_obstacles(matrix, object_position, object_dimension, resolution, origin)

    # Create the grid object
    grid = Grid(matrix=matrix)

    # Convert the start and end positions to grid coordinates
    start_coords = to_grid_coords(start_position, resolution, origin)
    end_coords = to_grid_coords(end_position, resolution, origin)
    print(f"start: {start_coords}, end: {end_coords}")

    # Create start and end nodes in the grid
    start = grid.node(*start_coords)
    end = grid.node(*end_coords)

    # Initialize the A* path planner
    planner = AStar(diagonal_movement=DMove.ALWAYS)

    # Run the A* algorithm to compute the path
    path, runs = planner.get_planned_path(start, end, grid)

    grid.visualize(path, start, end, save_to=r"/c/Users/wston/Desktop/Purdue/Robotic_Motion/DroneSim/Assets/RoboSim/bin/path.html")

    # Convert the path back to Unity world coordinates
    waypoints = [
        ((origin[0] + p.x * resolution).__round__(2), (origin[1] + p.y * resolution).__round__(2), (origin[2] + p.z * resolution).__round__(2))
        for p in path
    ]

    with open(waypoint_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(waypoints)