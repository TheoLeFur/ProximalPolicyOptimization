import numpy as np


class ReplayBuffer:

    def __init__(self, max_capacity: int = 1000000):

        self.max_capacity = max_capacity
        self.paths = []
        self.obs = None
        self.acs = None
        self.concatenated_rews = None
        self.unconcatenated_rews = None
        self.next_obs = None
        self.terminals = None

    def add_rollout(self, paths):
        for path in paths:
            self.paths.append(path)

        observations = np.concatenate([path["observation"] for path in paths])
        actions = np.concatenate([path["action"] for path in paths])
        next_observations = np.concatenate(
            [path["next_observation"] for path in paths])
        terminals = np.concatenate([path["terminal"] for path in paths])
        concatenated_rewards = np.concatenate(
            [path["reward"] for path in paths])
        unconcatenated_rewards = [path["reward"] for path in paths]

        if self.obs is None:

            self.obs = observations[-self.max_capacity:]
            self.acs = actions[-self.max_capacity:]
            self.next_obs = next_observations[-self.max_capacity:]
            self.terminals = terminals[-self.max_capacity:]
            self.concatenated_rews = concatenated_rewards[-self.max_capacity:]
            self.unconcatenated_rews = unconcatenated_rewards[-self.max_capacity:]

        else:

            self.obs = np.concatenate(
                [self.obs, observations])[-self.max_capacity:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_capacity:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_capacity:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_capacity:]
            self.concatenated_rews = np.concatenate(
                [self.concatenated_rews, concatenated_rewards]
            )[-self.max_capacity:]
            if isinstance(unconcatenated_rewards, list):
                self.unconcatenated_rews += unconcatenated_rewards
            else:
                self.unconcatenated_rews.append(unconcatenated_rewards)


        def sample_random_rollout(self, num_rollouts : int):
            random_indices = np.random.permutation(len(self.paths))[:num_rollouts]
            return self.paths[random_indices]
        
        def sample_recent_rollout(self, num_rollouts = 1):
            return self.paths[-num_rollouts:]


        def sample_random_data(self, batch_size : int):
            rand_indices = np.random.permutation(self.obs.shape[0])[:batch_size]
            return self.obs[rand_indices], self.acs[rand_indices], self.concatenated_rews[rand_indices], self.next_obs[rand_indices], self.terminals[rand_indices]


    
