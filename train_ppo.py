import numpy as np
import gym
import torch
import argparse
import matplotlib.pyplot as plt
from collections import OrderedDict
from agents.ppo_agent import PPOAgent
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter




def plot(datas):

    plt.plot(datas)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Datas')
    plt.show()

    print('Max :', np.max(datas))
    print('Min :', np.min(datas))
    print('Avg :', np.mean(datas))


def train_agent(train_params):


    writer = SummaryWriter("logs")

    env = gym.make(train_params["env"])
    N = train_params["N"]
    n_epochs = train_params["n_epochs"]
    learning_rate = train_params["learning_rate"]
    n_games = train_params["n_games"]


    params = {
        "action_dim": env.action_space.n,
        "observation_dim": env.observation_space.shape[0],
        "n_layers": 2,
        "size": 64,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "learning_rate": 3e-4,
        "discrete": True,
        "eps_clip": 0.2,
        "use_rnd_exploration" : False,
        "rnd_output_size" : 5,
        "rnd_size" :64,
        "rnd_n_layers" : 2,
        "ent_coeff" : 0.001
    }

    agent = PPOAgent(hparams=params)
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

                losses = agent.train()
                
                total_losses.append(losses["total_loss"])
                actor_losses.append(losses["actor_loss"])
                critic_losses.append(losses["critic_loss"])
                learn_iters += 1

            observation = observation_
            score_history.append(score)


        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

        print({
            "episode" : i,
            "score" : score,
            "average score" : avg_score,
            "best score" : best_score,
            "learning_steps" : learn_iters
        })

        writer.add_scalar("score", score, i)
        writer.add_scalar("average_score", avg_score, i)
        writer.add_scalar("best_score", best_score, i)
        


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

    









    