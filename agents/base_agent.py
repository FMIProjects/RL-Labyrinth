import pygame

from environment.env_renderer import EnvRenderer
from environment.base_env import BaseMazeEnv
import pickle
import os

class BaseAgent:
    """
    Abstract base class for reinforcement learning agents.
    """

    def __init__(self, env: BaseMazeEnv | None ,store_pickle_path: str = None, load_pickle_path: str = None):
        self.env = env
        self.store_pickle_path = store_pickle_path
        self.load_pickle_path = load_pickle_path

    def choose_action(self, state):
        raise NotImplementedError("BaseAgent class method choose_action() is abstract.")

    def train(self, episodes=10000):
        raise NotImplementedError("BaseAgent class method train() is abstract.")

    def serialize(self):
        """
        Attempts to pickle this agent.
        Will return an Exception if the pickle path is not provided.
        """
        if self.store_pickle_path is None:
            raise Exception("No store pickle path provided.")

        with open(self.store_pickle_path, "wb") as file:
            pickle.dump(self, file)

    def deserialize(self):
        """
        Attempts to unpickle this agent.
        Will return an Exception if the pickle path is not provided or if it does not exist.
        Will overwrite all fields of the current object.
        """
        if self.load_pickle_path is None:
            raise Exception("No load pickle path provided.")

        if not os.path.exists(self.load_pickle_path):
            raise Exception("Load pickle path provided does not exist.")

        with open(self.load_pickle_path, "rb") as file:
            agent = pickle.load(file)
            self.__dict__.update(agent.__dict__)

def test_agent(env: BaseMazeEnv, agent: BaseAgent, episodes=10,verbose = False,renderer_assets_dir_path = "./assets"):
    """
    Run a test agent for a given number of episodes.
    Render each step in the maze.
    """

    maze_renderer = EnvRenderer(env,assets_dir_path=renderer_assets_dir_path)

    agent.epsilon = 0.0
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

            if verbose:
                print(f"Choose action: {action}")
            state, reward, done, _ = maze_renderer.step(action)
            state = env.get_observation()
            total_reward += reward
