# run_maze.py
import time

from environment.base_env import BaseMazeEnv
from environment.env_renderer import EnvRenderer
import pygame
import sys

# This is for testing porpose
def main():

    # Create Environment
    env = BaseMazeEnv(width=20, height=20, num_keys=3,num_obstacles=0,peek_distance=2,distance_type="manhattan")
    obs = env.reset()

    env_renderer = EnvRenderer(maze_env=env,cell_size=30,fps=30)

    done = False
    while not done:
        env_renderer.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Sample a random action
        action = env.sample_action()

        print(f"Action: {action}")

        obs, reward, done, _ = env_renderer.step(action)
        # agent_pos,agent_view,goal_distance,keys_distances,obstacle_distances = obs
        #
        # print(f"Agent pos: {agent_pos}")
        # print(f"Agent view:\n {agent_view}")
        # print(f"Goal distance: {goal_distance}")
        # print(f"Keys distances: {keys_distances}")
        # print(f"Obstacle distances: {obstacle_distances}")
        # print(f"Reward: {reward}, Done: {done}")

    env.close()

if __name__ == "__main__":
    main()