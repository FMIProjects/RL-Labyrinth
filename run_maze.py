# run_maze.py

from environment.maze_env import MazeEnv
import pygame
import sys

# This is for testing porpose (There is no agent yet)
def main():

    # Create Environment
    env = MazeEnv(width=30, height=20, num_keys=3)
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
        obs, reward, done, _ = env.step(action)
        print(f"Reward: {reward}, Done: {done}")

    env.close()

if __name__ == "__main__":
    main()