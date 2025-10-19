import csv
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
from environment import ENV
from tqdm import tqdm
# np.set_printoptions(threshold=np.inf)

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 1500.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)                                                                               
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # This line is necessary for Windows

    snr = 20 # SNR (20, 24, 28, 32, 36, 40)
    height = 10000 # Height (100,1000,10000)
    actions = []
    for i in range(1700):
        actions.append(np.random.choice([0,1,3,4], p = [0.3, 0.3, 0.3, 0.1]))
    actions = np.array(actions)

    e = ENV(snr, height)
    
    # Generate sample data since no input data is provided
    # Create sample map data (1700 time steps, each with some state data)
    sample_map = []
    sample_speed = []
    for i in range(1700):
        # Generate random state data (you should replace this with your actual data)
        state_data = np.random.rand(4, 84, 84)  # Example: 4 channels of 84x84 data
        sample_map.append(state_data)
        # Generate random speed data
        sample_speed.append(np.random.uniform(10, 100))  # Speed between 10-100
    
    # Set the data in the environment
    e.over(sample_map, sample_speed)
    speed = e.speed
    obss = []
    rtgs = []

    print('Initializing dataset and rtgs:')
    for i in tqdm(range(1700)):
        #
        # Change your input data here
        #
        obss.append(e.import_state(i))
        rtgs.append(e.act(actions[i],i))
    rtgs = np.array(rtgs)
    done_idxs = np.array([67, 219, 368, 504, 686, 887, 1042, 1161, 1335, 1517, 1700])
    start_index = 0
    timesteps = np.zeros(1701, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1


    print('*******************')
    # print('obss',len(obss)) # (1700, 4, 84, 84)
    # print('actions',actions.shape) # (1700,)
    # print('rtgs',fake_rtgs.shape) # (1700,)
    # print('*******************')

    train_dataset = StateActionReturnDataset(obss, 90, actions, done_idxs, rtgs, timesteps)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                      n_layer=6, n_head=8, n_embd=128, model_type='reward_conditioned', max_timestep=max(timesteps))
    model = GPT(mconf)

    # initialize a trainer instance and kick off training
    # Reduced epochs from 20 to 1 for fastest testing
    tconf = TrainerConfig(max_epochs=1, batch_size=128, learning_rate=6e-4,
                          lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*30*3,
                          num_workers=4, seed=11111, model_type='reward_conditioned', game='Breakout', max_timestep=max(timesteps))
    trainer = Trainer(model, train_dataset, speed, None, snr, height, tconf)

    trainer.train()