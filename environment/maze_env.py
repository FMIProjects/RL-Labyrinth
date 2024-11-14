import gym
from gym import spaces
import numpy as np
import pygame
from .procedural_generator import generate_maze

class MazeEnv(gym.Env):
    """
    Custom environment for an RL agent navigating a procedurally generated maze.
    """

    def __init__(self, width=10, height=10, num_keys=3, cell_size=20):
        super(MazeEnv, self).__init__()

        # Maze configuration
        self.width = width
        self.height = height
        self.num_keys = num_keys
        self.keys_collected = 0
        self.cell_size = cell_size
        # Neighbour cells used for drawing the lines in the maze UP,RIGHT,DOWN,LEFT
        self.cell_neighbours = np.array([[[0,0,0,0] for _ in range(width)] for _ in range(height)])
        self.maze = None
        # Action Space (Up, Down, Left, Right)
        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right

        # Observation Space
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(height, width), dtype=np.int32
        )

        # Initial Agent position
        self.agent_pos = [0, 0]

        # Initial Goal position
        self.goal_pos = [width - 1, height - 1]

        # Pygame Setup
        self.screen = None
        self.clock = pygame.time.Clock()

    def reset(self):
        """
        Reset the environment to its initial state and return the initial observation.
        Randomize start and finish positions.
        """
        # Generate a new maze layout
        self.maze = generate_maze(self.width, self.height, self.num_keys)

        # Randomize start position (agent position)
        self.agent_pos = self.random_empty_cell()

        # Randomize finish position (goal position) different from start
        self.goal_pos = self.random_empty_cell(exclude=self.agent_pos)

        self.keys_collected = 0
        return self.get_observation()

    def random_empty_cell(self, exclude=None):
        """
        Helper function to find a random empty cell in the maze.
        Ensures the cell is different from the exclude position.
        """

        while True:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if self.maze[y, x] == 0:
                if exclude is None or [x, y] != exclude:
                    return [x, y]

    def step(self, action):
        """
        Apply action to move the agent in the maze.
        """
        x, y = self.agent_pos

        if action == 0 and y > 0:  # Up
            y -= 1
        elif action == 1 and y < self.height - 1:  # Down
            y += 1
        elif action == 2 and x > 0:  # Left
            x -= 1
        elif action == 3 and x < self.width - 1:  # Right
            x += 1

        # Check if the new position is valid (not an obstacle)
        if self.maze[y, x] != 1:
            self.agent_pos = [x, y]

        # Check if agent found a key
        if self.maze[y, x] == 2:
            self.keys_collected += 1
            self.maze[y, x] = 0

        # Check if agent reached the goal and has enough keys
        done = bool(self.maze[y, x] == 4 and self.keys_collected >= self.num_keys)

        # Calculate reward
        reward = 1 if done else -0.1  # Penalty for each move, reward for completion

        return self.get_observation(), reward, done, {}

    def get_observation(self):
        """
        Return the current state of the maze with the agent's position.
        """
        obs = self.maze.copy()
        obs[self.agent_pos[1], self.agent_pos[0]] = 3  # Agent's position marked as 3
        obs[self.goal_pos[1], self.goal_pos[0]] = 4  # Goal position marked as 4
        return obs

    def render(self, mode="human"):
        """
        Render the maze visually using Pygame.
        """
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width * self.cell_size, self.height * self.cell_size)
            )
            pygame.display.set_caption("Maze Environment")

        self.screen.fill((255, 255, 255))

        # Draw the maze
        for y in range(self.height):
            for x in range(self.width):
                cell_value = self.maze[y, x]
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                if cell_value == 1:    # Obstacle
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)
                elif cell_value == 2:  # Key
                    pygame.draw.circle(
                        self.screen, (255, 215, 0), rect.center, self.cell_size // 4
                    )
                elif cell_value == 4:  # Goal
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)

        # Draw the agent
        agent_rect = pygame.Rect(
            self.agent_pos[0] * self.cell_size,
            self.agent_pos[1] * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        pygame.draw.rect(self.screen, (0, 0, 255), agent_rect)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
