import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as T
from matplotlib import pyplot as plt
from collections import deque
from PIL import Image
import torch.optim as optim
from collections import namedtuple
from itertools import count
import math

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from utils.env_utils import get_screen
from models.DQN import DQN
from train.replay import ReplayMemory
from models.optimizer import optimize_model
from utils.action import select_action

def train_agent(args):
    # if gpu is to be used
    device = torch.device("cuda" 
        if torch.cuda.is_available() and args.ngpu > 0 
        else "cpu")

    # Build env (first level, right only)
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # setup networks
    init_screen = get_screen(env, device)
    _, _, screen_height, screen_width = init_screen.shape

    # Get number of actions from gym action space
    args.n_actions = env.action_space.n

    policy_net = DQN(screen_height, screen_width, args.n_actions).to(device)
    target_net = DQN(screen_height, screen_width, args.n_actions).to(device)

    if args.targetNet:
        target_net.load_state_dict(torch.load(args.targetNet, map_location=device))

    if args.policyNet:
        target_net.load_state_dict(torch.load(args.policyNet, map_location=device))


    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)


    args.steps_done = 0

    num_episodes = 1

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen(env, device)
        current_screen = get_screen(env, device)
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = select_action(state, policy_net, args, device)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env, device)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model(optimizer, memory, policy_net, target_net, args, device)
            if done:
                episode_durations.append(t + 1)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(policy_net.state_dict(), args.output_policyNet)
            torch.save(target_net.state_dict(), args.output_targetNet)

        if i_episode % 10 == 0:
            print(f'{i_episode+1}/{num_episodes}: Completed Episode.')

    print('Complete')
    env.close()

    torch.save(policy_net.state_dict(), args.output_policyNet)
    torch.save(target_net.state_dict(), args.output_targetNet)




    