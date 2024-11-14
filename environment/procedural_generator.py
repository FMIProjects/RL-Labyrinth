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


def generate_hunt_and_kill(height = 10,  width = 10):

    # Each cell is a neighbour of another cell if there is no wall between them
    # The order is:
    #   0->UP,
    #   1->RIGHT,
    #   2->DOWN,
    #   3->LEFT

    cell_neighbours = np.array([[[0, 0, 0, 0] for _ in range(width)] for _ in range(height)])
    maze = np.zeros((height,width),dtype=int)

    hunt_and_kill = True

    # Start from a random point
    row = np.random.randint(0, height)
    col = np.random.randint(0, width)


    while hunt_and_kill:

        # Compute the nearby cells
        up_cell = maze[row - 1, col] if row > 0 else 1
        down_cell = maze[row + 1, col] if row < height - 1 else 1
        left_cell = maze[row, col - 1] if col > 0 else 1
        right_cell = maze[row, col + 1] if col < width - 1 else 1

        # Mark the current cell visited
        maze[row, col] = 1

        # Kill phase
        # while there are still unvisited neighbours
        while not(up_cell == 1 and down_cell == 1 and left_cell == 1 and right_cell == 1):

            # Mark the current cell visited
            maze[row, col] = 1

            # Compute the valid(unvisited) neighbours
            valid_neighbour_cells = []

            if not up_cell:
                valid_neighbour_cells.append((-1,0))

            if not down_cell:
                valid_neighbour_cells.append((1,0))

            if not left_cell:
                valid_neighbour_cells.append((0,-1))

            if not right_cell:
                valid_neighbour_cells.append((0,1))

            # Get a random valid neighbour
            random_index = np.random.randint(0,len(valid_neighbour_cells))
            next_cell = valid_neighbour_cells[random_index]

            # Check which cell was chosen in order to mark correctly the cell neighbours

            # UP
            if next_cell == (-1,0):
                cell_neighbours[row,col,0] = 1
                next_row,next_col = row-1,col
                cell_neighbours[next_row, next_col, 2] = 1
            # DOWN
            elif next_cell == (1,0):
                cell_neighbours[row, col, 2] = 1
                next_row,next_col = row+1,col
                cell_neighbours[next_row, next_col, 0] = 1
            # LEFT
            elif next_cell == (0,-1):
                cell_neighbours[row, col, 3] = 1
                next_row,next_col = row,col-1
                cell_neighbours[next_row, next_col, 1] = 1
            # RIGHT
            elif next_cell == (0,1):
                cell_neighbours[row, col, 1] = 1
                next_row,next_col = row,col+1
                cell_neighbours[next_row, next_col, 3] = 1

            # Update the col and row indexes and the nearby cells
            col,row = next_col,next_row

            up_cell = maze[row - 1, col] if row > 0 else 1
            down_cell = maze[row + 1, col] if row < height - 1 else 1
            left_cell = maze[row, col - 1] if col > 0 else 1
            right_cell = maze[row, col + 1] if col < width - 1 else 1

        found = False

        # iterate through each row
        for i in range(height):

            if found:
                break

            for j in range(width):

                # If there is an unvisited cell that has a previously visited cell then make the 2 cells neighbours
                if maze[i, j] == 0:

                    #UP
                    if i > 0 and maze[i-1,j] == 1:
                        # Update the neighbours
                        cell_neighbours[i,j,0] = 1
                        cell_neighbours[i-1,j,2] = 1
                        # Update the new start point
                        row = i
                        col = j
                        # Mark as found and the hunt can stop
                        found = True
                        # Break from j loop
                        break

                    # RIGHT
                    elif j < width-1 and maze[i,j+1] == 1:
                        cell_neighbours[i,j,1] = 1
                        cell_neighbours[i,j+1,3] = 1
                        row = i
                        col = j
                        found = True
                        break

                    # DOWN
                    elif i < height-1 and maze[i+1,j] == 1:
                        cell_neighbours[i,j,2] = 1
                        cell_neighbours[i+1,j,0] = 1
                        row = i
                        col = j
                        found = True
                        break

                    # LEFT
                    elif j > 0 and maze[i,j-1] == 1:
                        cell_neighbours[i, j, 3] = 1
                        cell_neighbours[i,j-1, 1] = 1
                        row = i
                        col = j
                        found = True
                        break

        # if the matrix iteration has ended and there is no cell found then we can end the hunt and kill
        if not found:
            return maze, cell_neighbours




if __name__ == "__main__":
    maze, n = generate_hunt_and_kill(10, 10)
    print(maze)
    print(n)

