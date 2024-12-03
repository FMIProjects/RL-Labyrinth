# run_maze.py
import time

from environment.maze_env import MazeEnv
import pygame
import sys

# This is for testing porpose (There is no agent yet)
def main():

    # Create Environment
    env = MazeEnv(width=6, height=6,cell_size=30, num_keys=3,num_obstacles=0,peek_distance=2,distance_type="manhattan",fps=30)
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
        agent_pos,agent_view,goal_distance,keys_distances,obstacle_distances = obs

        print(f"Agent pos: {agent_pos}")
        print(f"Agent view:\n {agent_view}")
        print(f"Goal distance: {goal_distance}")
        print(f"Keys distances: {keys_distances}")
        print(f"Obstacle distances: {obstacle_distances}")
        print(f"Reward: {reward}, Done: {done}")

    env.close()

if __name__ == "__main__":
    main()