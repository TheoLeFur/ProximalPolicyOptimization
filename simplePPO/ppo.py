import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMemory:

    def __init__(self, batch_size):

        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):

        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
            np.array(self.actions),\
            np.array(self.probs),\
            np.array(self.vals),\
            np.array(self.rewards),\
            np.array(self.dones),\
            batches

    def store_memory(self, state, action, prob, val, reward, done):

        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):

        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []


class ActorNetwork(nn.Module):

    def __init__(self, n_actions, input_dims, learning_rate, size, checkpoint_dir="tmp/ppo"):

        super(ActorNetwork, self).__init__()

        self.checkpoint = os.path.join(checkpoint_dir, "actor_torch_ppo")
        self.actor = nn.Sequential(
            nn.Linear(input_dims, size),
            nn.Tanh(),
            nn.Linear(size, size),
            nn.Tanh(),
            nn.Linear(size, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.to(self.device)

    def forward(self, state):

        state = torch.Tensor(state).to(self.device)
        distribution = self.actor(state)
        distribution = Categorical(distribution)

        return distribution

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint))


class Critic(nn.Module):

    def __init__(self, input_dims, learning_rate, size, checkpoint_dir="tmp/ppo"):

        super(Critic, self).__init__()
        self.checkpoint = os.path.join(checkpoint_dir, "critic_torch_ppo")

        self.critic = nn.Sequential(
            nn.Linear(input_dims, size),
            nn.Tanh(),
            nn.Linear(size, size),
            nn.Tanh(),
            nn.Linear(size, 1),
        )

        self.optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.device = "mps" if torch.backends.mps.is_available else "cpu"
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint))


class Agent:

    def __init__(self, n_actions, input_dims, size = 64, gamma=0.99, learning_rate=3e-4, gae_lambda=0.95, policy_clip=0.1, batch_size=64, N=2048, n_epochs=10, ):

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, learning_rate, size)
        self.critic = Critic(input_dims, learning_rate, size)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):

        state = torch.Tensor(observation).to(self.actor.device)

        distribution = self.actor(observation)
        value = self.critic(state)
        action = distribution.sample()
        probs = distribution.log_prob(action).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return probs, action, value

    def train(self):

        for _ in range(self.n_epochs):

            states, actions, old_probs, vals, rewards, dones, batches = self.memory.generate_batches()

            advantage = np.zeros(len(rewards), dtype=np.float32)

            for t in range(len(rewards) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards) - 1):

                    a_t += discount * \
                        (rewards[k] + self.gamma*vals[k+1]) * \
                        (1-int(dones[k])) - vals[k]
                    discount *= self.gamma * self.gae_lambda

                advantage[t] = a_t
            advantage = torch.tensor(
                advantage, dtype=torch.float32, device=self.actor.device)
            values = torch.tensor(vals, dtype = torch.float32, device = self.actor.device)

            for batch in batches:

                print(states.shape)

                states = torch.tensor(
                    states[batch], dtype=torch.float32, device=self.actor.device)
                old_probs = torch.tensor(
                    old_probs[batch], dtype=torch.float32, device=self.actor.device)
                actions = torch.tensor(
                    actions[batch], dtype=torch.float32, device=self.actor.device)

                distribution = self.actor(states)
                critic_value = torch.squeeze(self.critic(states))
                new_probs = distribution.log_prob(actions)
                ratio = torch.exp(new_probs - old_probs)

                surr1 = advantage[batch] * ratio
                surr2 = torch.clamp(ratio, 1 - self.policy_clip,
                                    1 + self.policy_clip) * advantage[batch]

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss_fn = nn.MSELoss()
                critic_loss = critic_loss_fn(
                    advantage[batch] + values[batch], critic_value).mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
