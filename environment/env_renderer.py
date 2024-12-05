import pygame
import random

from environment.maze_env import MazeEnv

class EnvRenderer:
    def __init__(self, maze_env: MazeEnv, cell_size=20, fps=60):
        self.maze_env = maze_env
        self.cell_size = cell_size
        self.fps = fps

        # Pygame Setup
        self.screen = None
        self.clock = pygame.time.Clock()

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (
                    self.maze_env.width * self.cell_size,
                    self.maze_env.height * self.cell_size,
                )
            )
            pygame.display.set_caption("Maze Environment")

        # Assets loading

        self.agent_images = {
            self.maze_env.UP: pygame.transform.scale(
                pygame.image.load("assets/agent_up.png").convert_alpha(),
                (self.cell_size, self.cell_size),
            ),
            self.maze_env.RIGHT: pygame.transform.scale(
                pygame.image.load("assets/agent_right.png").convert_alpha(),
                (self.cell_size, self.cell_size),
            ),
            self.maze_env.DOWN: pygame.transform.scale(
                pygame.image.load("assets/agent_down.png").convert_alpha(),
                (self.cell_size, self.cell_size),
            ),
            self.maze_env.LEFT: pygame.transform.scale(
                pygame.image.load("assets/agent_left.png").convert_alpha(),
                (self.cell_size, self.cell_size),
            ),
        }

        self.agent_direction = self.maze_env.UP
        self.background_images = []

        self.ground_images = [
            pygame.image.load(f"assets/ground{i}.png").convert() for i in range(1, 12)
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

        self.key_image = pygame.transform.scale(
            self.key_image, (self.cell_size, self.cell_size)
        )
        self.door_image = pygame.transform.scale(
            self.door_image, (self.cell_size, self.cell_size)
        )
        self.red_door_image = pygame.transform.scale(
            self.red_door_image, (self.cell_size, self.cell_size)
        )
        self.obstacle_image = pygame.transform.scale(
            self.obstacle_image, (self.cell_size, self.cell_size)
        )

        # Fill background with random images
        for row in range(self.maze_env.height):
            for col in range(self.maze_env.width):
                x, y = col * self.cell_size, row * self.cell_size
                random_image = random.choice(self.ground_images)
                self.background_images.append(random_image)
                self.screen.blit(random_image, (x, y))

    def step(self, action):
        """
        Method needed in order to update the agent direction.
        """
        self.agent_direction = action
        return self.maze_env.step(action)

    def render(self, mode="human"):
        """
        Render the maze visually using Pygame.
        """

        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        BLUE = (0, 0, 255)
        GREEN = (0, 255, 0)
        YELLOW = (255, 215, 0)
        RED = (255, 0, 0)
        GREY = (128, 128, 128)

        # Draw background
        i = 0
        for row in range(self.maze_env.height):
            for col in range(self.maze_env.width):
                x, y = col * self.cell_size, row * self.cell_size
                self.screen.blit(self.background_images[i], (x, y))
                i += 1

        # Draw lines
        for row in range(self.maze_env.height):
            for col in range(self.maze_env.width):
                x, y = col * self.cell_size, row * self.cell_size

                # Get the bool values to check where to draw the walls
                up, right, down, left = self.maze_env.cell_neighbours[row, col]

                # Draw up line
                if not up or row == 0:
                    pygame.draw.line(
                        self.screen, WHITE, (x, y), (x + self.cell_size, y), 3
                    )
                # Draw right line
                if not right or col == self.maze_env.width - 1:
                    pygame.draw.line(
                        self.screen,
                        WHITE,
                        (x + self.cell_size, y),
                        (x + self.cell_size, y + self.cell_size),
                        3,
                    )
                # Draw down line
                if not down or row == self.maze_env.height - 1:
                    pygame.draw.line(
                        self.screen,
                        WHITE,
                        (x, y + self.cell_size),
                        (x + self.cell_size, y + self.cell_size),
                        3,
                    )
                # Draw left line
                if not left or col == 0:
                    pygame.draw.line(
                        self.screen, WHITE, (x, y), (x, y + self.cell_size), 3
                    )

        # Note: the position values of the agent,goal and keys are stored in the following format [y,x]

        # Draw the agent
        agent_image = self.agent_images[self.agent_direction]
        self.screen.blit(
            agent_image,
            (
                self.maze_env.agent_pos[1] * self.cell_size,
                self.maze_env.agent_pos[0] * self.cell_size,
            ),
        )

        # Draw the goal
        self.screen.blit(
            self.door_image,
            (
                self.maze_env.goal_pos[1] * self.cell_size,
                self.maze_env.goal_pos[0] * self.cell_size,
            ),
        )

        # Draw the agent and goal if they are both on the same cell and the keys are still not collected
        if self.maze_env.agent_pos == self.maze_env.goal_pos:
            self.screen.blit(
                self.red_door_image,
                (
                    self.maze_env.goal_pos[1] * self.cell_size,
                    self.maze_env.goal_pos[0] * self.cell_size,
                ),
            )

        # Draw the keys
        for key_pos in self.maze_env.keys_pos:
            self.screen.blit(
                self.key_image,
                (key_pos[1] * self.cell_size, key_pos[0] * self.cell_size),
            )

        # Draw the obstacles

        for obstacle_pos in self.maze_env.obstacles_pos:
            self.screen.blit(
                self.obstacle_image,
                (obstacle_pos[1] * self.cell_size, obstacle_pos[0] * self.cell_size),
            )

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None