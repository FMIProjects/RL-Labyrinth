import pygame

from environment.env_renderer import EnvRenderer
from environment.maze_env import MazeEnv


class BaseAgent:
    """
    Abstract base class for reinforcement learning agents.
    """
    def __init__(self,env: MazeEnv):
        self.env = env

    def choose_action(self,state):
        raise NotImplementedError("BaseAgent class method choose_action() is abstract.")

    def train(self,episodes=10000):
        raise NotImplementedError("BaseAgent class method train() is abstract.")



def test_agent(env: MazeEnv, agent: BaseAgent, episodes=10):
    """
    Run a test agent for a given number of episodes.
    Render each step in the maze.
    """

    maze_renderer = EnvRenderer(env)

    agent.epsilon = 0
    for episode in range(episodes):

        env.reset()
        state = env.get_observation()
        total_reward = 0
        done = False

        while not done:

            maze_renderer.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()


            action = agent.choose_action(state)
            print(f"Choose action: {action}")
            state, reward, done, _ = maze_renderer.step(action)
            state = env.get_observation()
            total_reward += reward