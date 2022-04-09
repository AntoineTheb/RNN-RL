import argparse
import gym
import numpy as np
import os
import pybullet_envs  # noqa F401
# import pybulletgym  # noqa F401 register PyBullet enviroments with open ai gym
import torch

from algos import DDPG, PPO, TD3
from utils import memory

from world import World
from generator import LaneVehicleGenerator
from metric import TravelTimeMetric
from environment import TSCEnv

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10, test=False):
    policy.eval_mode()
    avg_reward = 0.
    env = gym.make(env_name)
    env.seed(seed + 100)

    for _ in range(eval_episodes):
        if test:
            env.render(mode='human', close=False)
        state, done = env.reset(), False
        hidden = None
        while not done:
            if test:
                env.render(mode='human', close=False)
            action, hidden = policy.select_action(np.array(state), hidden)
            # env.render(mode='human', close=False)
            state, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    policy.train_mode()
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

class RNN_RL():

    def __init__(self, config_file, timesteps, episodes):
        # Policy name (TD3, DDPG or OurDDPG)
        self.policy = "TD3"
        # Threads
        self.threads = 2
        # Config File 
        self.config_file = config_file
        # Sets Gym, PyTorch and Numpy seeds
        self.seed = 0
        # Time steps initial random policy is used
        self.start_timesteps = 1e4
        # How often (time steps) we evaluate
        self.eval_freq= 10
        # Max time steps to run environment
        self.max_timesteps = 1e6
        # Std of Gaussian exploration noise
        self.expl_noise = 0.25
        # Batch size for both actor and critic
        self.batch_size = 100
        # Memory size
        self.memory_size = 1e6
        # Learning rate
        self.lr = 3e-4
        # Discount factor
        self.discount = 0.99
        # Target network update rate
        self.tau = 0.005
        # Noise added to target policy during critic update
        self.policy_noise = 0.25
        # Range to clip target policy noise
        self.noise_clip = 0.5
        # Frequency of delayed policy updates
        self.policy_freq = 2
        # Model width
        self.hidden_size = 256
        # Use recurrent policies or not
        self.recurrent = True
        # Save model and optimizer parameters
        self.save_model = True
        # Model load file name, "" doesn't load, "default" uses file_name
        self.load_model = ''
        # Don't train and just run the model
        self.test = False
        # Agents
        self.agents = []
        # World 
        self.world = World(config_file, thread_num=self.threads)
        # Metrics
        self.metric = TravelTimeMetric(self.world)
        # Steps
        self.steps = timesteps
        # State Dims
        self.state_dim = 28
        # Max actions
        self.max_action = 8
        # Recurrent Actor
        self.recurrent_actor = True
        # Recurrent Critic
        self.recurrent_critic = True
        # Episodes
        self.episodes = episodes
        self.save_rate = 20000
        self.save_dir = 'model/dqn'
        self.log_dir = 'log/dqn'
        self.action_interval = 20
        
        file_name = f"{self.policy}_{self.seed}"
        print("---------------------------------------")
        print(f"Policy: {self.policy}, Seed: {self.seed}")
        print("---------------------------------------")

        if not os.path.exists("./results"):
            os.makedirs("./results")

        if self.save_model and not os.path.exists("./models"):
            os.makedirs("./models")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if self.policy == "TD3":
            for i in self.world.intersections:
                self.agents.append(TD3.TD3(
                    # Action Dimension
                    gym.spaces.Discrete(len(i.phases)),
                    # Observation Generator
                    LaneVehicleGenerator(self.world, i, ["lane_count"], in_only=True, average=None),
                    # Reward Generator
                    LaneVehicleGenerator(self.world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
                    # ID
                    i,
                    # Max Action
                    self.max_action,
                    # Hidden Dims
                    self.hidden_size,
                    # Discount
                    self.discount,
                    # Tau
                    self.tau,
                    # Policy Noise
                    self.policy_noise * self.max_action,
                    # Noise Clip
                    self.noise_clip * self.max_action,
                    # Policy Freq
                    self.policy_freq,
                    # Learning Rate
                    3e-4,
                    # Recurrent Actor
                    self.recurrent_actor,
                    # Recurrent Critic
                    self.recurrent_critic,
                    # State Dimension
                    self.state_dim
                ))

        if self.load_model != "":
            policy_file = file_name \
                if self.load_model == "default" else self.load_model
            policy.load(f"{policy_file}")

        self.replay_buffer = memory.ReplayBuffer(
            self.state_dim, self.max_action , self.hidden_size,
            self.memory_size, recurrent=self.recurrent_actor)

        self.env = TSCEnv(self.world, self.agents, self.metric)

        
    def train(self):
        total_decision_num = 0
        for e in range(self.episodes):
            last_obs = self.env.reset()
            if e % self.save_rate == self.save_rate - 1:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file("replay_%s.txt" % e)
            else:
                self.env.eng.set_save_replay(False)
            episodes_rewards = [0 for i in self.agents]
            episodes_decision_num = 0
            i = 0
            while i < self.steps:
                if i % self.action_interval == 0:
                    actions = []
                    for agent_id, agent in enumerate(self.agents):
                        if total_decision_num > agent.learning_start:
                        #if True:
                            actions.append(agent.get_action(last_obs[agent_id]))
                        else:
                            actions.append(agent.sample())

                    rewards_list = []
                    for _ in range(self.action_interval):
                        obs, rewards, dones, _ = self.env.step(actions)
                        i += 1
                        rewards_list.append(rewards)
                    rewards = np.mean(rewards_list, axis=0)

                    for agent_id, agent in enumerate(self.agents):
                        agent.remember(last_obs[agent_id], actions[agent_id], rewards[agent_id], obs[agent_id])
                        episodes_rewards[agent_id] += rewards[agent_id]
                        episodes_decision_num += 1
                        total_decision_num += 1
                    
                    last_obs = obs

                for agent_id, agent in enumerate(self.agents):
                    if total_decision_num > agent.learning_start and total_decision_num % agent.update_model_freq == agent.update_model_freq - 1:
                        agent.replay()
                    if total_decision_num > agent.learning_start and total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                        agent.update_target_network()
                if all(dones):
                    break
            if e % self.save_rate == self.save_rate - 1:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                for agent in self.agents:
                    agent.save_model(self.save_dir)
            log_text = "{}, {}".format(e, self.env.eng.get_average_travel_time())
            for agent_id, agent in enumerate(self.agents):
                log_text += ", {}, {}".format(agent_id, episodes_rewards[agent_id] / episodes_decision_num)
            print(log_text)
                    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run RNN RL')
    parser.add_argument('--timesteps', type=int, default=3600, help='Time for run - 3600 is 1h')
    parser.add_argument('--episodes', type=int, default=100, help='Epoch steps')
    args = parser.parse_args()

    rnn_rl = RNN_RL('/notebooks/RNN-RL/config_dqn.json', args.timesteps, args.episodes)
    print("Created ENV")
    rnn_rl.train()
