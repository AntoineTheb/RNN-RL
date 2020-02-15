import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(
        self, state_dim, action_dim, hidden_size,
        max_size=int(5e3), recurrent=False
    ):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.recurrent = recurrent

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, action_dim))
        self.next_state = np.zeros((self.max_size, state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

        if self.recurrent:
            self.h = np.zeros((self.max_size, hidden_size))
            self.nh = np.zeros((self.max_size, hidden_size))

            self.c = np.zeros((self.max_size, hidden_size))
            self.nc = np.zeros((self.max_size, hidden_size))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def add(
        self, state, action, next_state, reward, done, hiddens, next_hiddens
    ):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        if self.recurrent:
            h, c = hiddens
            nh, nc = next_hiddens

            # Detach the hidden state so that BPTT only goes through 1 timestep
            self.h[self.ptr] = h.detach().cpu()
            self.c[self.ptr] = c.detach().cpu()
            self.nh[self.ptr] = nh.detach().cpu()
            self.nc[self.ptr] = nc.detach().cpu()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=100, shuffle=True):
        if not shuffle:
            ind = np.arange(0, self.size)
        else:
            ind = np.random.randint(0, self.size, size=int(batch_size))

        if self.recurrent:
            h = torch.tensor(self.h[ind][None, ...],
                             requires_grad=True,
                             dtype=torch.float).to(self.device)
            c = torch.tensor(self.c[ind][None, ...],
                             requires_grad=True,
                             dtype=torch.float).to(self.device)
            nh = torch.tensor(self.nh[ind][None, ...],
                              requires_grad=True,
                              dtype=torch.float).to(self.device)
            nc = torch.tensor(self.nc[ind][None, ...],
                              requires_grad=True,
                              dtype=torch.float).to(self.device)

            # TODO: Return hidden states or not
            hidden = (h, c)
            next_hidden = (nh, nc)

            s = torch.FloatTensor(
                self.state[ind][:, None, :]).to(self.device)
            a = torch.FloatTensor(
                self.action[ind][:, None, :]).to(self.device)
            ns = torch.FloatTensor(
                self.next_state[ind][:, None, :]).to(self.device)
            r = torch.FloatTensor(
                self.reward[ind][:, None, :]).to(self.device)
            d = torch.FloatTensor(
                self.not_done[ind][:, None, :]).to(self.device)

        else:
            hidden = None
            next_hidden = None

            s = torch.FloatTensor(self.state[ind]).to(self.device)
            a = torch.FloatTensor(self.action[ind]).to(self.device)
            ns = \
                torch.FloatTensor(self.next_state[ind]).to(self.device)
            r = torch.FloatTensor(self.reward[ind]).to(self.device)
            d = torch.FloatTensor(self.not_done[ind]).to(self.device)

        return s, a, ns, r, d, hidden, next_hidden

    def clear_memory(self):
        self.ptr = 0
        self.size = 0
