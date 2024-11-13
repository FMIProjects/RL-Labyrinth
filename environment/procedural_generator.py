# procedural_generator.py

import numpy as np

def generate_maze(width=10, height=10, num_keys=3):
    """
    Generates a maze as a 2D grid with randomly placed obstacles, keys, and a goal.

    Parameters:
        width (int): Width of the maze.
        height (int): Height of the maze.
        num_keys (int): Number of keys to place in the maze.

    Returns:
        np.ndarray: 2D array representing the maze.
    """
    # Initialize maze with empty spaces
    maze = np.zeros((height, width), dtype=int)

    # Add obstacles (marked as 1)
    # Here goes to procedural generator


    # Place keys (marked as 2)
    keys_placed = 0
    while keys_placed < num_keys:
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        if maze[y, x] == 0:
            maze[y, x] = 2  # 2 indicates a key
            keys_placed += 1

    # Place start (marked as 3) and goal (marked as 4)
    maze[0, 0] = 3                   # 3 indicates the starting point
    maze[height - 1, width - 1] = 4  # 4 indicates the goal

    return maze