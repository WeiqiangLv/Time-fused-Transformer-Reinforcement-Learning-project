import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize
import csv
import math
import os
import random
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import constants
from OTFS_modulator import Simulator


# My environment for drones communication
class ENV(object):
    def __init__(self, SNR, height):
        self.ber = []
        self.speed = []
        self.snr = SNR
        self.end_flag = False
        self.done = np.array([67, 219, 368, 504, 686, 887, 1042, 1161, 1335, 1517, 1700])
        self.length = np.array([67, 152, 149, 136, 182, 201, 155, 119, 174, 182, 183])
        self.height = height
        #
        # Change your input data here
        #
        self.map = None

    def reset_game(self, time):
        return

    def over(self, data, ss):
        self.map = data
        self.speed = ss
        return

    def act(self, action_number, time):
        if action_number == 0 or action_number == 2:
            # Keep old action
            return 0

        elif action_number == 1:
            a = Simulator()
            b = a.run(self.snr, self.speed[time], 'OFDM', 32, self.height)
            self.ber += [b[0]] # BER
            return 0 - int(math.log(b[1])) 
        elif action_number == 3:
            a = Simulator()
            b = a.run(self.snr, self.speed[time], 'OTFS', 32, self.height)
            self.ber += [b[0]] # BER
            return -1 - int(math.log(b[1])) # DIS

        elif action_number == 4:
            a = Simulator()
            b = a.run(self.snr, self.speed[time], 'OTFS', 64, self.height)
            self.ber += [b[0]] # BER
            return -2 - int(math.log(b[1])) # DIS

    def game_over(self, time):
        return time in self.done

    def import_state(self, time):
        #
        # Import your input data here
        #
        return self.map[time]

    def update_state(self, action, time):
        #
        # Update state based on action and time
        # For now, just return the state at the given time
        #
        return self.map[time]

    # Gaus function
    def gaus(self, mu, sigma, no):
        return int(np.random.normal(mu, sigma, no))
