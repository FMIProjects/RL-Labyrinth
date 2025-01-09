import numpy as np
from tqdm import tqdm
from environment.base_env import BaseMazeEnv
from agents.base_agent import BaseAgent
from visualiser import plot_rewards_and_lengths

class SarsaAgent(BaseAgent):
    """
    General SARSA Agent for MazeEnv using ε-greedy strategy to choose an action.
    """

    def __init__(
            self,
            env: BaseMazeEnv | None = None,
            alpha=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.1,
            load_pickle_path=None,
            store_pickle_path=None,
            episodes_trained=0,
    ):
        super().__init__(env,store_pickle_path=store_pickle_path,load_pickle_path=load_pickle_path)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Probability to explore a new path
        self.epsilon_decay = (
            epsilon_decay  # Rate of decay of the probability to explore a new path
        )
        self.epsilon_min = epsilon_min  # Minimum probability to explore a new path
        self.q_table = {}  # Q(s, a)
        self.episodes_trained = episodes_trained

    def choose_action(self, state):
        """
        ε-greedy strategy

        With ε probability choose either to explore a random path or to exploit the best current path.
        """

        # Explore a new path (choose a random action)
        if np.random.rand() < self.epsilon:
            return self.env.sample_action()

        # Exploit the best path (return argmax(Q(s,a)))
        return max(
            range(self.env.action_space.n),
            key=lambda a: self.q_table.get((state, a), 0),
        )

    def update_q_value(self, state, action, reward, next_state, next_action):
        """
        Update a Q-value entry using SARSA formula.
        Q(s,a) = Q(s,a) + α * (reward + γ * Q(s',a') - Q(s,a))
        """

        current_q = self.q_table.get((state, action), 0)
        next_q = self.q_table.get((next_state, next_action), 0)
        temporal_difference_value = reward + self.gamma * next_q
        temporal_difference_error = temporal_difference_value - current_q
        self.q_table[(state, action)] = (
                current_q + self.alpha * temporal_difference_error
        )

    def train(self, episodes=10000):
        """
        Trains the agent in a number of episodes.
        In the end stores a .pkl file of the Q(s,a) values if requested.
        """
        rewards = []  # List to store rewards from each episode
        episode_lengths = []  # List to store the number of steps per episode
        
        # Loop through each episode
        for episode in tqdm(range(episodes)):
            self.episodes_trained += 1
            self.env.reset()

            # Get the current state and choose an action
            current_state = self.env.get_observation()
            current_action = self.choose_action(current_state)

            total_reward = 0
            steps_taken = 0
            done = False

            while not done:
                # Get the next state reward and choose the next action
                next_state, reward, done, _ = self.env.step(current_action)
                next_action = self.choose_action(next_state)

                # Update the q value
                self.update_q_value(
                    current_state, current_action, reward, next_state, next_action
                )

                # Update state and action for next step
                current_state = next_state
                current_action = next_action
                total_reward += reward
                steps_taken += 1  # Track the number of steps

            # After each episode reduce the exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Append the total reward and episode length to the lists
            rewards.append(total_reward)
            episode_lengths.append(steps_taken)

        # Plot rewards and episode lengths after all episodes are done
        plot_rewards_and_lengths(rewards, episode_lengths, episodes)

        print("Training finished!")
