import matplotlib.pyplot as plt
import numpy as np


def plot_test_rewards(rewards, episodes, filename='test_rewards.png'):
    """
    Plot the rewards obtained in the test episodes.
    """
    plt.figure(figsize=(10, 6))

    # Plot the rewards for each test episode
    plt.scatter(range(1, episodes + 1), rewards, color='r', label='Test Rewards', marker='o')
    
    # Adding labels and title
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'Agent Test Performance Over {episodes} Episodes')
    plt.grid(True)
    plt.legend()

    plt.savefig(filename)
    plt.close()

    print(f"Test performance plot saved as {filename}")

def plot_rewards_and_lengths(rewards, episode_lengths, episodes):
    """
    Plots both rewards and episode lengths over episodes.
    """
    import matplotlib.pyplot as plt
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Rewards', color='b')
    ax1.plot(range(1, episodes + 1), rewards, color='b', label='Rewards')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Episode Lengths (Steps)', color='g')
    ax2.plot(range(1, episodes + 1), episode_lengths, color='g', label='Episode Lengths')
    ax2.tick_params(axis='y', labelcolor='g')
    
    plt.title('Agent Performance: Rewards & Episode Lengths Over Episodes')
    plt.grid(True)
    
    plt.savefig('training_performance.png')
    plt.close()

    print("Performance plot saved as 'training_performance.png'")