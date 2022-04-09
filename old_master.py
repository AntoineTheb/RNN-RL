import argparse
import gym
import numpy as np
import os
import pybullet_envs  # noqa F401
# import pybulletgym  # noqa F401 register PyBullet enviroments with open ai gym
import torch

from algos import DDPG, PPO, TD3OLD
from utils import memory


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


def main():
    parser = argparse.ArgumentParser()
    # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--policy", default="TD3")
    # OpenAI gym environment name
    parser.add_argument("--env", default="HopperBulletEnv-v0")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=0, type=int)
    # Time steps initial random policy is used
    parser.add_argument("--start_timesteps", default=1e4, type=int)
    # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=5e3, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.25)
    # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=100, type=int)
    # Memory size
    parser.add_argument("--memory_size", default=1e6, type=int)
    # Learning rate
    parser.add_argument("--lr", default=3e-4, type=float)
    # Discount factor
    parser.add_argument("--discount", default=0.99)
    # Target network update rate
    parser.add_argument("--tau", default=0.005)
    # Noise added to target policy during critic update
    parser.add_argument("--policy_noise", default=0.25)
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.5)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=2, type=int)
    # Model width
    parser.add_argument("--hidden_size", default=256, type=int)
    # Use recurrent policies or not
    parser.add_argument("--recurrent", action="store_true")
    # Save model and optimizer parameters
    parser.add_argument("--save_model", action="store_true")
    # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model", default="")
    # Don't train and just run the model
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    print("TYPE STATE DIM {}".format(type(state_dim)))
    print(state_dim)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # TODO: Add this to parameters
    recurrent_actor = args.recurrent
    recurrent_critic = args.recurrent

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "hidden_dim": args.hidden_size,
        "discount": args.discount,
        "tau": args.tau,
        "recurrent_actor": recurrent_actor,
        "recurrent_critic": recurrent_critic,
    }

    def mixAndMatch(*args, **kwargs):
        print(f' Args: {args}' )
        print(f' Kwargs: {kwargs}' )
    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        mixAndMatch(**kwargs)
        policy = TD3OLD.TD3(**kwargs)

    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    elif args.policy == "PPO":
        # TODO: Add kwargs for PPO
        kwargs["K_epochs"] = 10
        kwargs["eps_clip"] = 0.1
        policy = PPO.PPO(**kwargs)
        args.start_timesteps = 0
        n_update = 2048

    if args.load_model != "":
        policy_file = file_name \
            if args.load_model == "default" else args.load_model
        policy.load(f"{policy_file}")

    if args.test:
        eval_policy(policy, args.env, args.seed, eval_episodes=10, test=True)
        return

    replay_buffer = memory.ReplayBuffer(
        state_dim, action_dim, args.hidden_size,
        args.memory_size, recurrent=recurrent_actor)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    best_reward = evaluations[-1]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    hidden = policy.get_initial_states()

    for t in range(1, int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
            _, next_hidden = policy.select_action(np.array(state), hidden)
        else:
            a, next_hidden = policy.select_action(np.array(state), hidden)
            action = (
                a + np.random.normal(
                    0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)

        done_bool = float(
            done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(
            state, action, next_state, reward, done_bool, hidden, next_hidden)

        state = next_state
        hidden = next_hidden
        episode_reward += reward

        # Train agent after collecting sufficient data
        if (not policy.on_policy) and t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)
        elif policy.on_policy and t % n_update == 0:
            policy.train(replay_buffer)
            replay_buffer.clear_memory()

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it
            #  will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} "
                f"Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            hidden = policy.get_initial_states()

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            if evaluations[-1] > best_reward and args.save_model:
                policy.save(f"./models/{file_name}")

            np.save(f"./results/{file_name}", evaluations)


if __name__ == "__main__":
    main()
