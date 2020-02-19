# Recurrent Reinforcement Learning in Pytorch
Experiments with reinforcement learning and recurrent neural networks

Disclaimer: My code is very much based on Scott Fujimotos's [TD3 implementation](https://github.com/sfujim/TD3)

TODO: Cite properly

## Motivations
This repo serves as a exercise for myself to properly understand what goes into using RNNs with Deep Reinforcement Learning

[1: _Kapturowski et al. 2019_](https://openreview.net/pdf?id=r1lyTjAqYX) provides insight on how RL algorithms might use memory while training. 
For on-policy algorithms such as PPO, it makes sense to train on whole trajectories and discard the RNN's memory. However, could the hidden state at each timestep be kept, and each timestep used as an independant "batch" item ?

For off-policy algorithms, such as DDPG, things get a bit more complicated. The naive option of training on whole trajectories is not computationally desirable, especially if enforcing a specific trajectory length is not an option. Another option would be to train on timesteps without using the RNN's memory. However, this implies losing the advantages associated with using RNNs.

An other option would be to keep the hidden state of the RNN associated with each timestep. However, the hidden states will become "outdated" as the timestep stay in memory and the network learns a new internal representation. [1] also suggests allowing the network a "burn-in" period by saving n timesteps and letting the network make it down hidden state before training on the timestep.

## Questions I have and want answers for:
- Is training on full trajectories necessary ?
- If not, how should timesteps and hidden states be batched ?
- Does the critic need to predict at each timestep to have its own hidden state ?
- Could the hidden state even be discarded ?
- With Actor-Critic methods, do they both have to be recurrent ?
- Are all these questions dependant on the algorithm used ?

## Goals
- Implement deep RL algorithms in a simple and understandable way
- Acquire a better understanding of how recurrence impacts DRL algorithms

## TODOs
- Hidden state handling strategies
- More algorithms (A2C, R2D1, BCQ, REINFORCE, etc.)
- Comments, lots of them
- Results
- Cleanup

## Implementations
- TD3
- DDPG
- PPO (needs tweaking)

## How to install
```
$ pip install -r requirements.txt`
```

## Results

![](/plots/HopperBulletEnv-v0.png)
  
