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
        self.action_dim = action_dim

        if self.recurrent:
            self.l1 = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        else:
            self.l1 = nn.Linear(state_dim, hidden_dim)

        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        self.max_action = max_action
        self.policy_noise = policy_noise

    def forward(self, state, hidden):
        if self.recurrent:
            self.l1.flatten_parameters()
            p, h = self.l1(state, hidden)
        else:
            p, h = torch.tanh(self.l1(state)), None

        p = torch.tanh(self.l2(p.data))
        return p, h

    def act(self, state, hidden):
        p, h = self.forward(state, hidden)
        action = torch.tanh(self.actor(p))

        return action * self.max_action, h

    def evaluate(self, state, action, hidden):
        p, h = self.forward(state, hidden)
        action_mean, _ = self.act(state, hidden)

        cov_mat = torch.eye(self.action_dim).to(device) * self.policy_noise

        dist = MultivariateNormal(action_mean, cov_mat)
        _ = dist.sample()
        action_logprob = dist.log_prob(action)
        entropy = dist.entropy()
        values = self.critic(p)

        if self.recurrent:
            values = values[..., 0]
        else:
            action_logprob = action_logprob[..., None]

        return values, action_logprob, entropy


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
        eps_clip=.2,
        lmbda=0.95,
        lr=3e-4,
        K_epochs=80,
        recurrent_actor=False,
        recurrent_critic=False,
    ):
        self.on_policy = True
        self.recurrent = recurrent_actor
        self.actorcritic = ActorCritic(
            state_dim, action_dim, hidden_dim, max_action, policy_noise,
            is_recurrent=recurrent_actor
        ).to(device)
        self.target = copy.deepcopy(self.actorcritic)
        self.optimizer = torch.optim.Adam(self.target.parameters())

        self.discount = discount
        self.lmbda = lmbda
        self.tau = tau
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.actor_loss_coeff = 1.
        self.critic_loss_coeff = 0.74
        self.entropy_loss_coeff = 0.01

    def get_initial_states(self):
        h_0, c_0 = None, None
        if self.actorcritic.recurrent:
            h_0 = torch.zeros((
                self.actorcritic.l1.num_layers,
                1,
                self.actorcritic.l1.hidden_size),
                dtype=torch.float)
            h_0 = h_0.to(device=device)

            c_0 = torch.zeros((
                self.actorcritic.l1.num_layers,
                1,
                self.actorcritic.l1.hidden_size),
                dtype=torch.float)
            c_0 = c_0.to(device=device)
        return (h_0, c_0)

    def select_action(self, state, hidden):
        if self.recurrent:
            state = torch.FloatTensor(
                state.reshape(1, -1)).to(device)[:, None, :]
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        action, hidden = self.actorcritic.act(state, hidden)
        return action.cpu().data.numpy().flatten(), hidden

    def train(self, replay_buffer):

        # Sample replay buffer
        state, action, next_state, reward, not_done, hidden, next_hidden = \
            replay_buffer.on_policy_sample()

        running_actor_loss = 0
        running_critic_loss = 0

        discounted_reward = 0
        rewards = []

        for r, is_terminal in zip(reversed(reward), reversed(1 - not_done)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = r + (self.discount * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = rewards[..., None]

        # log_prob of pi(a|s)
        _, prob_a, _ = self.actorcritic.evaluate(
            state,
            action,
            hidden)

        # TODO: PPO Update
        # PPO allows for multiple gradient steps on the same data
        for _ in range(self.K_epochs):

            # V_pi'(s) and pi'(a|s)
            v_s, logprob, dist_entropy = self.target.evaluate(
                state,
                action,
                hidden)

            assert rewards.size() == v_s.size(), \
                '{}, {}'.format(rewards.size(), v_s.size())
            # Finding Surrogate Loss:
            advantages = rewards - v_s

            # Ratio between probabilities of action according to policy and
            # target policies

            assert logprob.size() == prob_a.size(), \
                '{}, {}'.format(logprob.size(), prob_a.size())
            ratio = torch.exp(logprob - prob_a)

            # Surrogate policy loss
            assert ratio.size() == advantages.size(), \
                '{}, {}'.format(ratio.size(), advantages.size())

            surrogate_policy_loss_1 = ratio * advantages
            surrogate_policy_loss_2 = torch.clamp(
                ratio,
                1-self.eps_clip,
                1+self.eps_clip) * advantages
            # PPO "pessimistic" policy loss
            actor_loss = -torch.min(
                surrogate_policy_loss_1,
                surrogate_policy_loss_2)

            # Surrogate critic loss: MSE between "true" rewards and prediction
            # TODO: Investigate size mismatch
            assert(v_s.size() == rewards.size())

            surrogate_critic_loss_1 = F.mse_loss(
                v_s,
                rewards)
            surrogate_critic_loss_2 = torch.clamp(
                surrogate_critic_loss_1,
                -self.eps_clip,
                self.eps_clip
            )
            # PPO "pessimistic" critic loss
            critic_loss = torch.max(
                surrogate_critic_loss_1,
                surrogate_critic_loss_2)

            # Entropy "loss" to promote entropy in the policy
            entropy_loss = dist_entropy[..., None].mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss = ((critic_loss * self.critic_loss_coeff) +
                    (self.actor_loss_coeff * actor_loss) -
                    (entropy_loss * self.entropy_loss_coeff))
            # print(loss.size(), loss)
            loss.mean().backward(retain_graph=True)
            # print([p.grad for p in self.target.parameters()])
            nn.utils.clip_grad_norm_(self.target.parameters(),
                                     0.5)
            self.optimizer.step()

            # Keep track of losses
            running_actor_loss += actor_loss.mean().cpu().detach().numpy()
            running_critic_loss += critic_loss.mean().cpu().detach().numpy()

        self.actorcritic.load_state_dict(self.target.state_dict())
        torch.cuda.empty_cache()

    def save(self, filename):
        torch.save(self.actorcritic.state_dict(), filename)
        torch.save(self.optimizer.state_dict(),
                   filename + "_optimizer")

    def load(self, filename):
        self.actorcritic.load_state_dict(torch.load(filename))
        self.optimizer.load_state_dict(
            torch.load(filename + "_optimizer"))

    def eval_mode(self):
        self.actorcritic.eval()

    def train_mode(self):
        self.actorcritic.train()
