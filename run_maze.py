# run_maze.py
import time

from environment.maze_env import MazeEnv
import pygame
import sys

# This is for testing porpose (There is no agent yet)
def main():

    # Create Environment
    env = MazeEnv(width=20, height=20, num_keys=5)
    obs = env.reset()

    done = False
    while not done:
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Sample a random action
        action = env.action_space.sample()

        print(f"Action: {action}")

        obs, reward, done, _ = env.step(action)
        print(f"Reward: {reward}, Done: {done}")

    env.close()

if __name__ == "__main__":
    main()