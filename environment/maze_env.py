from pickletools import uint8

import gym
from gym import spaces
import numpy as np
import pygame
from .procedural_generator import generate_maze,maze_scale_up
class MazeEnv(gym.Env):
    """
    Custom environment for an RL agent navigating a procedurally generated maze.
    """

    def __init__(self, width=10, height=10, num_keys=3, cell_size=20):

        assert width % 2 == 0, "Maze width must be even."
        assert height % 2 == 0, "Maze height must be even."
        assert num_keys <= width * height, "Too many keys for current maze configuration."
        assert width > 0 and height > 0 and num_keys >= 0 and cell_size > 0, "Parameters values must be positive"

        super(MazeEnv, self).__init__()

        # Maze configuration
        self.width = width
        self.height = height
        self.num_keys = num_keys
        self.num_obstacles = 0
        self.keys_collected = 0
        self.cell_size = cell_size

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

        # Pygame Setup
        self.screen = None
        self.clock = pygame.time.Clock()

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

        # Randomize the key positions so that each is unique and different from the previously generated elements
        self.keys_pos = []
        for _ in range(self.num_keys):
            self.keys_pos.append(self.random_empty_cell(exclude=[self.agent_pos, self.goal_pos] + self.keys_pos))

        # TODO find a way to generate the obstacles without breaking the agent

        # Place in the maze the elements

        self.maze[self.agent_pos[0],self.agent_pos[1]] = self.AGENT
        self.maze[self.goal_pos[0], self.goal_pos[1]] = self.GOAL

        for key_pos in self.keys_pos:
            self.maze[key_pos[0], key_pos[1]] = self.KEY

        self.keys_collected = 0
        return self.get_observation()

    def random_empty_cell(self, exclude=None):
        """
        Helper function to find a random empty cell in the maze.
        Ensures the cell is different from the exclude positions.
        """

        while True:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if self.maze[y, x] == 0:
                if exclude is None or [x, y] not in exclude:
                    return [x, y]

    def step(self, action):
        """
        Apply action to move the agent in the maze.
        """

        row,col = self.agent_pos
        neighbours = self.cell_neighbours[row,col]

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

        # Check if agent reached the goal and has enough keys
        done = bool(self.agent_pos == self.goal_pos and self.keys_collected >= self.num_keys)

        # mark the agent state on the cell or the goal and agent state
        self.maze[row, col] = self.AGENT_AND_GOAL if row == self.agent_pos == self.goal_pos else self.AGENT

        # Calculate reward
        reward = 1 if done else -0.1  # Penalty for each move, reward for completion

        return self.get_observation(), reward, done, {}

    def get_observation(self):
        """
        Return the current state of the maze with the agent's position.
        """

        return self.maze

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

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width * self.cell_size, self.height * self.cell_size)
            )
            pygame.display.set_caption("Maze Environment")

        self.screen.fill(WHITE)

        for row in range(self.height):
            for col in range(self.width):
                x, y = col * self.cell_size, row * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)

                # Get the bool values to check where to draw the walls
                up, right, down, left = self.cell_neighbours[row, col]

                # Draw up line
                if not up or row == 0:
                    pygame.draw.line(self.screen, BLACK, (x, y), (x + self.cell_size, y), 1)
                # Draw right line
                if not right or col == self.width - 1:
                    pygame.draw.line(self.screen, BLACK, (x + self.cell_size, y), (x + self.cell_size, y + self.cell_size), 1)
                # Draw down line
                if not down or row == self.height - 1:
                    pygame.draw.line(self.screen, BLACK, (x, y + self.cell_size), (x + self.cell_size, y + self.cell_size), 1)
                # Draw left line
                if not left or col == 0:
                    pygame.draw.line(self.screen, BLACK, (x, y), (x, y + self.cell_size), 1)

        # Note: the position values of the agent,goal and keys are stored in the following format [y,x]

        # Draw the agent
        agent_rect = pygame.Rect(
            self.agent_pos[1] * self.cell_size,
            self.agent_pos[0] * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        pygame.draw.rect(self.screen, BLUE, agent_rect)

        # Draw the goal

        goal_rect = pygame.Rect(
            self.goal_pos[1] * self.cell_size,
            self.goal_pos[0] * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        pygame.draw.rect(self.screen, GREEN, goal_rect)

        # Draw the agent and goal if they are both on the same cell and the keys are still not collected
        if self.agent_pos == self.goal_pos:
            pygame.draw.rect(self.screen, RED, goal_rect)

        # Draw the keys

        for key_pos in self.keys_pos:
            key_rect = pygame.Rect(
                key_pos[1] * self.cell_size,
                key_pos[0] * self.cell_size,
                self.cell_size,
                self.cell_size,
            )
            pygame.draw.circle(
                self.screen, YELLOW, key_rect.center, self.cell_size // 4
            )


        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
