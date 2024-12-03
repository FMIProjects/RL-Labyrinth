# run_maze.py
import time

from environment.maze_env import MazeEnv
import pygame
import sys

# This is for testing porpose (There is no agent yet)
def main():

    # Create Environment
    env = MazeEnv(width=6, height=6,cell_size=30, num_keys=3,num_obstacles=2,distance_type="manhattan",fps=30)
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
        _,goal_distance,keys_distances,obstacle_distances = env.get_observation()

        print(f"Goal distance: {goal_distance}")
        print(f"Keys distances: {keys_distances}")
        print(f"Obstacle distances: {obstacle_distances}")
        print(f"Reward: {reward}, Done: {done}")

    env.close()

if __name__ == "__main__":
    main()