
import numpy as np

def generate_maze(height = 10, width = 10):

    """

    Generates a maze as a 2D array of [bool,bool,bool,bool] elements each bool standing for
    UP, RIGHT, DOWN, LEFT in this order using Hunt and Kill algorithm.
    Ex: if cells[i][j][1] = True it means that the right cell is not separated by a wall of the current cell

    """

    # Each cell is a neighbour of another cell if there is no wall between them
    # The order is:
    #   0->UP,
    #   1->RIGHT,
    #   2->DOWN,
    #   3->LEFT

    cell_neighbours = np.array([[[0, 0, 0, 0] for _ in range(width)] for _ in range(height)])
    maze = np.zeros((height,width),dtype=int)

    the_hunt_is_on = True

    # Start from a random point
    row = np.random.randint(0, height)
    col = np.random.randint(0, width)


    while the_hunt_is_on:

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
            return cell_neighbours


def maze_scale_up(cell_neighbours):

    """

    Scales up by 4 times an already generated maze and returns its 2d matrix of [bool,bool,bool,bool]

    Parameters:
        cell_neighbours: np.ndarray of bool of shape (n,m,4)

    Returns:
        np.ndarray of bool of shape (n*2,m*2,4)
    """

    height,width,_ = cell_neighbours.shape
    new_cell_neighbours = np.array([[[0, 0, 0, 0] for _ in range(width * 2)] for _ in range(height * 2)])

    for i in range(height):
        for j in range(width):

            # the up left cell keeps the neighbourhood of its parent cell on UP and LEFT
            new_cell_neighbours[i*2][j*2][0] = cell_neighbours[i][j][0]
            new_cell_neighbours[i*2][j*2][3] = cell_neighbours[i][j][3]
            # the other directions will be connected to the scaled up cells
            new_cell_neighbours[i*2][j*2][2] = 1
            new_cell_neighbours[i*2][j*2][1] = 1

            # UP RIGHT - same principle
            new_cell_neighbours[i*2][j*2+1][0] = cell_neighbours[i][j][0]
            new_cell_neighbours[i*2][j*2+1][1] = cell_neighbours[i][j][1]
            new_cell_neighbours[i*2][j*2+1][2] = 1
            new_cell_neighbours[i*2][j*2+1][3] = 1

            # DOWN RIGHT
            new_cell_neighbours[i*2+1][j*2+1][1] = cell_neighbours[i][j][1]
            new_cell_neighbours[i*2+1][j*2+1][2] = cell_neighbours[i][j][2]
            new_cell_neighbours[i*2+1][j*2+1][3] = 1
            new_cell_neighbours[i*2+1][j*2+1][0] = 1

            # DOWN LEFT
            new_cell_neighbours[i*2+1][j*2][2] = cell_neighbours[i][j][2]
            new_cell_neighbours[i*2+1][j*2][3] = cell_neighbours[i][j][3]
            new_cell_neighbours[i*2+1][j*2][0] = 1
            new_cell_neighbours[i*2+1][j*2][1] = 1

    return new_cell_neighbours


if __name__ == "__main__":

    import pygame
    import numpy as np

    pygame.init()

    rows, cols = 20,20
    cell_size = 40
    connections = generate_maze(rows, cols)
    connections = maze_scale_up(connections)
    rows *= 2
    cols *= 2
    width, height = cols * cell_size, rows * cell_size

    print(connections)

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Matrice Pygame Grid cu Conexiuni")

    def draw_grid_with_connections(connections):
        screen.fill(WHITE)

        for row in range(rows):
            for col in range(cols):
                x, y = col * cell_size, row * cell_size
                rect = pygame.Rect(x, y, cell_size, cell_size)

                up, right, down, left = connections[row, col]

                if not up or row == 0:
                    pygame.draw.line(screen, BLACK, (x, y), (x + cell_size, y), 1)

                if not right or col == cols - 1:
                    pygame.draw.line(screen, BLACK, (x + cell_size, y), (x + cell_size, y + cell_size), 1)

                if not down or row == rows - 1:
                    pygame.draw.line(screen, BLACK, (x, y + cell_size), (x + cell_size, y + cell_size), 1)

                if not left or col == 0:
                    pygame.draw.line(screen, BLACK, (x, y), (x, y + cell_size), 1)


    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw_grid_with_connections(connections)

        pygame.display.flip()

    pygame.quit()


