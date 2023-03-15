import numpy as np
import gym
import torch
import argparse
import matplotlib.pyplot as plt
from collections import OrderedDict
from agents.ppo_agent import PPOAgent
from tqdm import tqdm
from gym.envs.registration import register
from gym.envs.registration import registry



def train_agent(train_params):


    register(
            id='PointmassHard-v0',
            entry_point='ProximalPolicyOptimization.envs.pointmass::Pointmass',
            kwargs={'difficulty': 2}
            )
    env = gym.make('PointmassHard-v0')
    env.set_logdir(os.path.join("logs", "test"))
    N = train_params["N"]
    n_epochs = train_params["n_epochs"]
    learning_rate = train_params["learning_rate"]
    n_games = train_params["n_games"]

    params = {
        "action_dim": env.action_space.n,
        "observation_dim": env.observation_space.shape[0],
        "n_layers": 2,
        "size": 128,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "learning_rate": 3e-4,
        "discrete": True,
        "eps_clip": 0.2,
        "rnd_output_size" : 128,
        "rnd_size" :64,
        "rnd_n_layers" : 2
    }

    agent = PPOAgent(hparams=params)
    logs = OrderedDict()
    score_history = []
    best_score = env.reward_range[0]
    total_losses = []
    actor_losses = []
    critic_losses = []
    


    for i in tqdm(range(n_games)):
    
        observation, _ = env.reset()
        done = False
        learn_iters = 0
        n_steps = 0
        score = 0
        avg_score = 0
        

        while not done:

            action, prob, value = agent.get_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            score += reward
            n_steps += 1
            agent.add_to_replay_buffer(observation, action, prob, value, reward, done)

            if n_steps % N == 0:

                print(" ------- policy is updating - ------")
                losses = agent.train()
                
                total_losses.append(losses["total_loss"])
                actor_losses.append(losses["actor_loss"])
                critic_losses.append(losses["critic_loss"])
                learn_iters += 1

            observation = observation_
            score_history.append(score)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Proximal Policy Optimization")

    parser.add_argument("env", type = str)
    parser.add_argument("N", type = int, default = 20)
    parser.add_argument("n_epochs", type = int, default = 5)
    parser.add_argument("learning_rate", type = float, default = 3e-4)
    parser.add_argument("n_games", type = int, default = 500)


    args = parser.parse_args()
    params = vars(args)

    train_agent(params)

    









    