import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Reference implementations:
# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py
# https://github.com/seungeunrho/minimalRL/blob/master/ppo-lstm.py


class ActorCritic(nn.Module):
    def __init__(
        self, state_dim, action_dim, hidden_dim, max_action,
        policy_noise, is_recurrent=True
    ):
        super(ActorCritic, self).__init__()
        self.recurrent = is_recurrent

        if self.recurrent:
            self.l1 = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        else:
            self.l1 = nn.Linear(state_dim, hidden_dim)

        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        self.max_action = max_action
        self.policy_noise = policy_noise
        self.action_var = \
            torch.full((action_dim,), policy_noise*policy_noise).to(device)

    def forward(self, state, hidden):
        if self.recurrent:
            self.l1.flatten_parameters()
            p, h = self.l1(state, hidden)
        else:
            p, h = F.relu(self.l1(state)), None

        p = F.relu(self.l2(p))
        return p, h

    def act(self, state, hidden, test=True):
        p, h = self.forward(state, hidden)
        action_mean = self.actor(p)

        cov_mat = torch.diag(self.action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        if test:
            return action_mean, action_logprob, h
        else:
            return action, action_logprob, h

    def evaluate(self, state, action, hidden):
        p, h = self.forward(state, hidden)
        action_mean = self.actor(p)

        cov_mat = torch.diag(self.action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        _ = dist.sample()
        action_logprob = dist.log_prob(action)
        entropy = dist.entropy()
        values = self.critic(p)
        return action_logprob, values, entropy


class PPO(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        hidden_dim,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        eps_clip=1.0,
        lmbda=0.98,
        lr=3e-4,
        recurrent_actor=False,
        recurrent_critic=False,
    ):
        self.on_policy = True
        self.recurrent = recurrent_actor
        self.actorcritic = ActorCritic(
            state_dim, action_dim, hidden_dim, max_action, policy_noise,
            is_recurrent=recurrent_actor
        ).to(device)
        self.actorcritic_target = copy.deepcopy(self.actor)
        self.optimizer = torch.optim.Adam(self.actor.parameters())

        self.discount = discount
        self.lmbda = lmbda
        self.tau = tau
        self.eps_clip = eps_clip

    def get_initial_states(self):
        h_0, c_0 = None, None
        if self.actor.recurrent:
            h_0 = torch.zeros((
                self.actor.l1.num_layers,
                1,
                self.actor.l1.hidden_size),
                dtype=torch.float)
            h_0 = h_0.to(device=device)

            c_0 = torch.zeros((
                self.actor.l1.num_layers,
                1,
                self.actor.l1.hidden_size),
                dtype=torch.float)
            c_0 = c_0.to(device=device)
        return (h_0, c_0)

    def select_action(self, state, hidden):
        if self.recurrent:
            state = torch.FloatTensor(
                state.reshape(1, -1)).to(device)[:, None, :]
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        action, hidden = self.actor(state, hidden)
        return action.cpu().data.numpy().flatten(), hidden

    def train(self, replay_buffer):

        # Sample replay buffer
        state, action, next_state, reward, not_done, hidden, next_hidden = \
            replay_buffer.sample()

        # TODO: PPO Update
        pass

    def save(self, filename):
        torch.save(self.actorcritic.state_dict(), filename)
        torch.save(self.actorcritic.state_dict(),
                   filename + "_optimizer")

    def load(self, filename):
        self.actorcritic.load_state_dict(torch.load(filename))
        self.optimizer.load_state_dict(
            torch.load(filename + "_optimizer"))

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def train_mode(self):
        self.actor.train()
        self.critic.train()
