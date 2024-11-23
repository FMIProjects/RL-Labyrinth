from pickletools import uint8

import random
import gym
from gym import spaces
import numpy as np
import pygame
from .procedural_generator import generate_maze,maze_scale_up
from .distances import euclidean_distance,manhattan_distance

class MazeEnv(gym.Env):
    """
    Custom environment for an RL agent navigating a procedurally generated maze.
    """

    def __init__(self, width=10, height=10, num_keys=3, cell_size=20,distance_type = "manhattan", fps=30):

        # Macros
        self.AGENT = 1
        self.GOAL = 2
        self.KEY = 3
        self.OBSTACLE = 4
        self.AGENT_AND_GOAL = 5

        self.UP = 0
        self.RIGHT = 1
        self.DOWN = 2
        self.LEFT = 3

        assert width % 2 == 0, "Maze width must be even."
        assert height % 2 == 0, "Maze height must be even."
        assert num_keys <= width * height, "Too many keys for current maze configuration."
        assert width > 0 and height > 0 and num_keys >= 0 and cell_size > 0, "Parameters values must be positive"
        assert distance_type == "euclidean" or distance_type == "manhattan", "Distance type must be either 'euclidian' or 'manhattan'"

        super(MazeEnv, self).__init__()

        # Maze configuration
        self.width = width
        self.height = height
        self.num_keys = num_keys
        self.num_obstacles = 0
        self.keys_collected = 0
        self.cell_size = cell_size
        self.fps = fps

        # Neighbour cells used for drawing the lines in the maze UP,RIGHT,DOWN,LEFT
        self.cell_neighbours = maze_scale_up(generate_maze(self.height // 2, self.width // 2))
        self.maze = np.zeros((self.height, self.width),dtype=int)

        # Action Space (Up, Right, Down, Left)
        self.action_space = spaces.Discrete(4)

        # Observation Space
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(height, width), dtype=np.int32
        )

        # Note: All positions are stored as [row_index,col_index], pygame represents its matrices as [col_index,row_index]

        # Initial Agent position
        self.agent_pos = [0, 0]

        # Initial Goal position
        self.goal_pos = [width - 1, height - 1]

        # Initial key positions
        self.keys_pos = None

        # Initial obstacle positions
        self.obstacles_pos = None

        # Distances from goals and obstacles

        self.distance_type = distance_type

        self.goal_distance = -1
        self.keys_distances = []
        self.obstacles_distances = []

        # Pygame Setup
        self.screen = None
        self.clock = pygame.time.Clock()

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width * self.cell_size, self.height * self.cell_size)
            )
            pygame.display.set_caption("Maze Environment")

        # Assets loading

        self.agent_images = {
        self.UP: pygame.transform.scale(pygame.image.load("assets/agent_up.png").convert_alpha(), (self.cell_size, self.cell_size)),
        self.RIGHT: pygame.transform.scale(pygame.image.load("assets/agent_right.png").convert_alpha(), (self.cell_size, self.cell_size)),
        self.DOWN: pygame.transform.scale(pygame.image.load("assets/agent_down.png").convert_alpha(), (self.cell_size, self.cell_size)),
        self.LEFT: pygame.transform.scale(pygame.image.load("assets/agent_left.png").convert_alpha(), (self.cell_size, self.cell_size))
        }

        self.agent_direction = self.UP
        self.background_images = []

        self.ground_images = [
            pygame.image.load(f"assets/ground{i}.png").convert()
            for i in range(1, 12)            
            ]

        self.key_image = pygame.image.load(f"assets/key.png").convert_alpha()
        self.door_image = pygame.image.load(f"assets/door.png").convert_alpha()
        self.red_door_image = pygame.image.load(f"assets/red_door.png").convert_alpha()
        self.obstacle_image = pygame.image.load(f"assets/obstacle.png").convert_alpha()

        # Scale images to cell size
        self.ground_images = [
                pygame.transform.scale(img, (self.cell_size, self.cell_size))
                for img in self.ground_images
            ]

        self.key_image = pygame.transform.scale(self.key_image, (self.cell_size, self.cell_size))
        self.door_image = pygame.transform.scale(self.door_image, (self.cell_size, self.cell_size))
        self.red_door_image = pygame.transform.scale(self.red_door_image, (self.cell_size, self.cell_size))
        self.obstacle_image = pygame.transform.scale(self.obstacle_image, (self.cell_size, self.cell_size))

        # Fill background with random images
        for row in range(self.height):
            for col in range(self.width):
                x, y = col * self.cell_size, row * self.cell_size
                random_image = random.choice(self.ground_images)
                self.background_images.append(random_image)
                self.screen.blit(random_image, (x, y))

    def reset(self):
        """
        Reset the environment to its initial state and return the initial observation.
        Randomize start and finish positions.
        """
        # Generate a new scaled up maze layout
        self.cell_neighbours = maze_scale_up(generate_maze(self.height // 2, self.width // 2))
        self.maze = np.zeros((self.height, self.width))

        # Randomize start position (agent position)
        self.agent_pos = self.random_empty_cell()

        # Randomize finish position (goal position) different from start
        self.goal_pos = self.random_empty_cell(exclude=[self.agent_pos])

        # Compute goal distance
        self.compute_goal_distance()

        # Generate obstacles
        self.obstacles_pos = []
        self.num_obstacles = (self.width * self.height) // 128
        self.distribute_obstacles(self.num_obstacles)

        # Compute the obstacle distances
        self.compute_obstacle_distances()

        # Randomize the key positions so that each is unique and different from the previously generated elements
        self.keys_pos = []
        for _ in range(self.num_keys):
            random_key_pos = self.random_empty_cell(exclude=[self.agent_pos, self.goal_pos] + self.keys_pos + self.obstacles_pos)
            self.keys_pos.append(random_key_pos)

        # Compute keys distances
        self.compute_keys_distances()

        # Place in the maze the elements
        self.maze[self.agent_pos[0],self.agent_pos[1]] = self.AGENT
        self.maze[self.goal_pos[0], self.goal_pos[1]] = self.GOAL

        for key_pos in self.keys_pos:
            self.maze[key_pos[0], key_pos[1]] = self.KEY

        self.keys_collected = 0
        return self.get_observation()

    def distribute_obstacles(self,num_obstacles):

        """
        Distribute a number of maximum num_obstacles on the maze.
        The higher the obstacles the better the change.
        Iterate through 4x4 cell blocks and choose whether to place an obstacle in one of the 4.
        If the 4x4 cell block is chosen then choose again a random 1x1 cell to place an obstacle if that cell is not occupied.
        """

        # Get the total number of cells and calculate the probability of chosing a 4x4 cell block
        num_cells = (self.height * self.width) // 4
        probability = num_obstacles / num_cells

        for i in range(0,self.height,2):
            for j in range(0,self.width,2):

                coin_flip = np.random.rand()
                # Continue if the current cell is not chosen
                if coin_flip >= probability:
                    continue

                # Choose a random 1x1 cell to place the obstacle if the cell is not occupied by the agent or goal
                positions = [[i,j],[i,j+1],[i+1,j],[i+1,j+1]]
                random_index = np.random.randint(4)

                while positions[random_index] == self.agent_pos or positions[random_index] == self.goal_pos:
                    random_index = np.random.randint(4)

                # Mark the cell
                chosen_pos = positions[random_index]
                self.obstacles_pos.append(chosen_pos)
                self.maze[chosen_pos[0],chosen_pos[1]] = self.OBSTACLE
                num_obstacles -= 1


    def random_empty_cell(self, exclude=None):
        """
        Helper function to find a random empty cell in the maze.
        Ensures the cell is different from the exclude positions.
        """

        while True:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if self.maze[y, x] == 0:
                if exclude is None or [y, x] not in exclude:
                    return [y, x]

    def step(self, action):
        """
        Apply action to move the agent in the maze.
        """

        row,col = self.agent_pos
        neighbours = self.cell_neighbours[row,col]

        self.agent_direction = action
        # Check if the agent can go in chosen direction and update its position if possible

        if action == self.UP and neighbours[self.UP] == 1:
            row -= 1

        elif action == self.RIGHT and neighbours[self.RIGHT] == 1:
            col += 1

        elif action == self.DOWN and neighbours[self.DOWN] == 1:
            row += 1

        elif action == self.LEFT and neighbours[self.LEFT] == 1:
            col -= 1

        self.agent_pos = [row,col]

        # Check if the agent is in a key position and collect it ig so
        if self.maze[row, col] == self.KEY:
            self.keys_collected += 1
            self.maze[row, col] = 0
            self.keys_pos.remove(self.agent_pos)

        # Check if agent reached the goal and has enough keys or wheather the agent has fallen into an obstacle
        done = bool(self.agent_pos == self.goal_pos and self.keys_collected >= self.num_keys or self.agent_pos in self.obstacles_pos)

        # mark the agent state on the cell or the goal and agent state
        self.maze[row, col] = self.AGENT_AND_GOAL if row == self.agent_pos == self.goal_pos else self.AGENT

        # Calculate reward
        reward = -0.5 if done and self.agent_pos in self.obstacles_pos else 1 if done else 0  # Penalty for each move, reward for completion

        # Recompute distances
        self.compute_goal_distance()
        self.compute_obstacle_distances()
        self.compute_keys_distances()

        return self.get_observation(), reward, done, {}

    def compute_goal_distance(self):
        """
        Compute the distance between the agent and the goal based on the distance type.
        """
        if self.distance_type == "manhattan":
            self.goal_distance = manhattan_distance(self.agent_pos, self.goal_pos)
        elif self.distance_type == "euclidean":
            self.goal_distance = euclidean_distance(self.agent_pos, self.goal_pos)

    def compute_keys_distances(self):
        """
        Reset the keys distances list and recompute the distances from the agent to all keys.
        """
        self.keys_distances.clear()

        for key_pos in self.keys_pos:

            if self.distance_type == "euclidean":
                self.keys_distances.append(euclidean_distance(self.agent_pos, key_pos))
            elif self.distance_type == "manhattan":
                self.keys_distances.append(manhattan_distance(self.agent_pos, key_pos))

    def compute_obstacle_distances(self):
        """
        Reset the obstacle distances list and recompute the distances from the agent to all obstacles.
        """
        self.obstacles_distances.clear()

        for obstacle_pos in self.obstacles_pos:

            if self.distance_type == "euclidean":
                self.obstacles_distances.append(euclidean_distance(self.agent_pos, obstacle_pos))
            elif self.distance_type == "manhattan":
                self.obstacles_distances.append(manhattan_distance(self.agent_pos, obstacle_pos))

    def get_observation(self):
        """
        Return the current state of the maze with the maze configuration, goal distance, keys distances and obstacle distances.
        """

        return self.maze,self.goal_distance,self.keys_distances,self.obstacles_distances

    def render(self, mode="human"):
        """
        Render the maze visually using Pygame.
        """

        WHITE = (255, 255, 255)
        BLACK = (0,0,0)
        BLUE = (0, 0, 255)
        GREEN = (0, 255, 0)
        YELLOW = (255, 215, 0)
        RED = (255, 0, 0)
        GREY = (128,128,128)
        
        # Draw background
        i = 0
        for row in range(self.height):
            for col in range(self.width):
                x, y = col * self.cell_size, row * self.cell_size
                self.screen.blit(self.background_images[i], (x, y))
                i+=1
        
        # Draw lines
        for row in range(self.height):
            for col in range(self.width):
                x, y = col * self.cell_size, row * self.cell_size

                # Get the bool values to check where to draw the walls
                up, right, down, left = self.cell_neighbours[row, col]

                # Draw up line
                if not up or row == 0:
                    pygame.draw.line(self.screen, WHITE, (x, y), (x + self.cell_size, y), 3)
                # Draw right line
                if not right or col == self.width - 1:
                    pygame.draw.line(self.screen, WHITE, (x + self.cell_size, y), (x + self.cell_size, y + self.cell_size), 3)
                # Draw down line
                if not down or row == self.height - 1:
                    pygame.draw.line(self.screen, WHITE, (x, y + self.cell_size), (x + self.cell_size, y + self.cell_size), 3)
                # Draw left line
                if not left or col == 0:
                    pygame.draw.line(self.screen, WHITE, (x, y), (x, y + self.cell_size), 3)

        # Note: the position values of the agent,goal and keys are stored in the following format [y,x]

        # Draw the agent
        agent_image = self.agent_images[self.agent_direction]
        self.screen.blit(agent_image, (self.agent_pos[1] * self.cell_size, self.agent_pos[0] * self.cell_size))

        # Draw the goal
        self.screen.blit(self.door_image, (self.goal_pos[1]*self.cell_size, self.goal_pos[0]*self.cell_size))

        # Draw the agent and goal if they are both on the same cell and the keys are still not collected
        if self.agent_pos == self.goal_pos:
            self.screen.blit(self.red_door_image, (self.goal_pos[1]*self.cell_size, self.goal_pos[0]*self.cell_size))
        
        # Draw the keys
        for key_pos in self.keys_pos:
            self.screen.blit(self.key_image, (key_pos[1]*self.cell_size, key_pos[0]*self.cell_size))

        # Draw the obstacles

        for obstacle_pos in self.obstacles_pos:
            self.screen.blit(self.obstacle_image, (obstacle_pos[1]*self.cell_size, obstacle_pos[0]*self.cell_size))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None