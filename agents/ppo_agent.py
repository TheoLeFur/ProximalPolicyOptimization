import torch
import torch.nn as nn
import numpy as np

from infrastructure.replay_buffer import ReplayBuffer
from agents.base_agent import BaseAgent
from policies.MLPPolicy import PPOPolicy
from critics.PPOCritic import PPOCritic
from collections import OrderedDict


class PPOAgent(BaseAgent):

    def __init__(self, env, hparams):

        super(PPOAgent, self).__init__()

        self.env = env
        self.agent_params = hparams
        self.gamma = hparams["gamma"]
        self.num_critic_updates_per_agent_update = hparams[
            'num_critic_updates_per_agent_update']
        self.num_actor_updates_per_agent_update = hparams['num_actor_updates_per_agent_update']
        self.device = hparams["device"]

        self.actor = PPOPolicy(
            hparams["ac_dim"],
            hparams["ob_dim"],
            hparams["n_layers"],
            hparams["size"],
            hparams["device"],
            hparams["discrete"],
            hparams["eps_clip"]
        ).to(self.device)

        self.old_policy = PPOPolicy(
            hparams["ac_dim"],
            hparams["ob_dim"],
            hparams["n_layers"],
            hparams["size"],
            hparams["device"],
            hparams["discrete"],
            hparams["eps_clip"]
        ).to(self.device)

        self.critic = PPOCritic(hparams)
        self.old_policy.load_state_dict(self.actor.state_dict())
        self.replay_buffer = ReplayBuffer()

    def estimate_advantage(self, ob_no, next_ob_no, rew_no, terminal_n):

        ob, next_ob, rew, done = map(lambda x: torch.from_numpy(x).to(
            self.device), [ob_no, next_ob_no, rew_no, terminal_n])

        value = self.critic(ob)
        next_value = self.critic(next_ob) * (1 - done)
        advantage = rew + self.gamma * next_value - value

        return advantage.cpu().detach().numpy()

    def train(self, ob_no, ac_no, next_ob_no, rew_no, terminal_n):

        loss = OrderedDict()

        for critic_update in range(self.num_critic_updates_per_agent_update):
            loss["critic_update"] = self.critic.update(
                ob_no, ac_no, next_ob_no, rew_no, terminal_n)

        advantage = self.estimate_advantage(
            ob_no, next_ob_no, rew_no, terminal_n)
        old_log_prob = self.old_policy(ob_no)

        for actor_update in range(self.num_actor_updates_per_agent_update):
            loss["actor_update"] = self.actor.update(ob_no, ac_no, advantage, old_log_prob)
        self.old_policy.load_state_dict(self.actor.state_dict())

    def learn(self):
