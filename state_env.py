# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:03:19 2017

environment of the state transfer
reward: 1/(1+e^(30(0.9-x)))
initial state: 100000

@author: pc
"""

import numpy as np
from scipy.linalg import expm
import itertools
import configparser

def state_fidelity(state):
    return np.asarray((abs(state[-1]) ** 2)[0, 0])  # calculate fidelity

def calc_ipr(state):
    nh = np.shape(state)[0]
    ipr = 0
    for i in range(nh):
        ipr += np.real(state[i]*np.conjugate(state[i]))**2
    return np.real(1/ipr[0,0])

# reward function
def full_reward(next_state, prop, time_step):

    max_time = config.getint("system_parameters", "max_t_steps")
    fidelity_evolution = state_fidelity(next_state)

    for step in range(time_step, max_time):
        next_state = prop * next_state  # do operation
        fidelity_evolution = np.append(fidelity_evolution,
                                       state_fidelity(next_state))

    max_fid = np.max(fidelity_evolution)

    # reward function
    if max_fid < 0.8:
        reward = max_fid * 10

    if max_fid >= 0.8 and max_fid <= 0.95:
        reward = 100 / (1 + np.exp(10 * (0.95 - max_fid)))

    if max_fid > 0.95:
        reward = 2500 * max_fid

    reward = reward * (0.95**time_step)
    # a discount is given with respected to step

    return reward

def full_reward_zero(next_state, prop, time_step):

    max_time = config.getint("system_parameters", "max_t_steps")
    fidelity_evolution = state_fidelity(next_state)
    
    for step in range(time_step, max_time):
        next_state = prop0 * next_state  # do operation
        fidelity_evolution = np.append(fidelity_evolution,
                                       state_fidelity(next_state))

    max_fid = np.max(fidelity_evolution)

    # reward function
    if max_fid < 0.8:
        reward = max_fid * 10

    if max_fid >= 0.8 and max_fid <= 0.95:
        reward = 100 / (1 + np.exp(10 * (0.95 - max_fid)))

    if max_fid > 0.95:
        reward = 2500 * max_fid

    reward = reward * (0.95**time_step)
    # a discount is given with respected to step

    return reward

def original_reward(next_state, prop, time_step):
    
    fidelity = state_fidelity(next_state)
    
    if fidelity < 0.8:
        reward = fidelity*10
    if fidelity >= 0.8 and fidelity <= 0.95:
        reward = 100/(1+np.exp(10*(0.95-fidelity)))
    if fidelity > 0.95:
        reward = 2500


    reward = reward*(0.95**time_step)
        ############# a discount is given with respected to step
    return reward

def ipr_reward(next_state, prop, time_step):
    
    ipr = calc_ipr(next_state)
    fidelity = state_fidelity(next_state)
    reward = fidelity/ipr
    reward = reward*(0.95**time_step)
    return reward

# read parameters from config file  
config_file = "config.ini" 
config = configparser.ConfigParser()
config.read(config_file)

chain_length = config.getint("system_parameters", "chain_length")
field_strength = config.getfloat("system_parameters", "field_strength")
coupling = config.getfloat("system_parameters", "coupling")
DT = config.getfloat("system_parameters", "tstep_length")
action_set = config.get("system_parameters", "action_set")
reward_function = config.get("learning_parameters", "reward_function")



    
if action_set == "zhang":
    nc = 3  # number of control sites, nc=3,there are totally 16 actions.
    # defining action, each row of 'mag' corresponds to one allowed configuration
    def binact(A):  # action label
        m = np.zeros(nc)
        for ii in range(
            nc
        ):  # transfer action to a binary list, for example: action=5, x=[1, 0, 1, 0], the first and third magnetic is on
            m[nc - 1 - ii] = A >= 2 ** (nc - 1 - ii)
            A = A - 2 ** (nc - 1 - ii) * m[nc - 1 - ii]
        return m

    mag = []
    for ii in range(8):  # control at the beginning
        mag.append(list(np.concatenate((binact(ii) * field_strength, np.zeros(chain_length - nc)))))

    for ii in range(1, 8):  # control at the end
        mag.append(list(np.concatenate((np.zeros(chain_length - nc), binact(ii) * field_strength))))

    mag.append([field_strength for ii in range(chain_length)])

    action_hamiltonians = np.zeros((16, chain_length, chain_length))

    for actions in mag:
        ham = (
                np.diag([coupling for i in range(chain_length - 1)], 1) * (1 - 0j)
                + np.diag([coupling for i in range(chain_length - 1)], -1) * (1 + 0j)
                + np.diag(actions)
            )
        action_hamiltonians[mag.index(actions)] = ham

elif action_set == "oaps":

    def one_field_actions(bmax, chain_length, coupling):
        """
        Generates a set of action matrices corresponding to fields acting on every individual
        site.

        Parameters:
        bmax (float): The maximum value of the field, used in diagonal elements.
        chain_length (int): Chain length, which defines the size of action matrices.

        Returns:
        numpy.ndarray: A 3D numpy array of shape (chain_length + 1, chain_length, chain_length) representing the action matrices.
        """

        action_matrices = np.zeros((chain_length + 1, chain_length, chain_length))
        J = 1.0

        for i in range(0, chain_length):

            for k in range(0, chain_length - 1):
                action_matrices[i + 1, k, k + 1] = J
                action_matrices[i + 1, k + 1, k] = action_matrices[i + 1, k, k + 1]

            action_matrices[i + 1, i, i] = bmax

        for k in range(0, chain_length - 1):
            action_matrices[0, k, k + 1] = J
            action_matrices[0, k + 1, k] = action_matrices[0, k, k + 1]

        return action_matrices

    action_hamiltonians = one_field_actions(field_strength,chain_length, coupling)

if reward_function == "full reward":
    reward_function = full_reward
elif reward_function == "original":
    reward_function = original_reward
elif reward_function == "ipr":
    reward_function = ipr_reward
elif reward_function == "full reward zero":
    reward_function = full_reward_zero
    prop0 = expm(-1j * action_hamiltonians[0] * DT)

class State(object):
    def __init__(self):
        super(State, self)
        self.action_space = action_hamiltonians
        print(self.action_space)
        self.n_actions = config.getint("system_parameters","n_actions")  # allowed action number =16
        self.n_features = config.getint("learning_parameters","number_of_features")  # the dimension of input vector
        self.stp = 0  # initially at the first step
        self.stmax = config.getint(
            "system_parameters", "max_t_steps"
        )  # maximum allowed steps

        self.maxfid = 0  # initialize the maximum fidelity

    def reset(self):
        psi = [0 for i in range(chain_length)]  # initial state is [1;0;0;0;0...]
        psi[0] = 1
        self.state = np.array([str(i) for i in psi])
        self.state = np.array(list(itertools.chain(*[(i.real, i.imag) for i in psi])))
        self.stp = 0
        self.maxfid = 0

        return self.state
    
    def random_reset(self):
        
        psi = np.random.rand(chain_length) + 1j * np.random.rand(chain_length)  # random complex numbers
        psi = psi / np.linalg.norm(psi)  # normalize the state
        self.state = np.array([str(i) for i in psi])
        self.state = np.array(list(itertools.chain(*[(i.real, i.imag) for i in psi])))
        self.stp = 0
        self.maxfid = 0
        
        return self.state

    def step(self, actionnum):

        self.stp += 1

        actions = self.action_space[actionnum]  # magnetic field configuration

        ham = actions
        
        prop = expm(-1j * ham * DT)  # evolution operator

        statess = [
            complex(self.state[2 * i], self.state[2 * i + 1]) 
            for i in range(chain_length)
        ]  # transfer real vector to complex vector

        statelist = np.transpose(np.mat(statess))  # to 'matrix'
        next_state = prop * statelist  # do operation
    

        reward = reward_function(next_state, prop, self.stp)
        fidelity = state_fidelity(next_state)
        doned = False
        
        if fidelity > self.maxfid:
            self.maxfid = state_fidelity(next_state)
        if fidelity > 0.95:
            doned = True

        next_states = [next_state[i, 0]
                       for i in range(chain_length)]  # 'matrix' to list
        next_states = np.array(
            list(itertools.chain(*[(i.real, i.imag) for i in next_states]))
        )  # complex to real vector

        self.state = next_states  # this vector is input to the network
        return next_states, reward, doned, fidelity

    def noisy_step(self, actionnum):

        self.stp += 1

        actions = self.action_space[actionnum]  # magnetic field configuration

        ham = actions
        
        prop = expm(-1j * ham * DT)  # evolution operator

        statess = [
            complex(self.state[2 * i], self.state[2 * i + 1]) 
            for i in range(chain_length)
        ]  # transfer real vector to complex vector

        statelist = np.transpose(np.mat(statess))  # to 'matrix'
        next_state = prop * statelist  # do operation

        next_state = next_state + 0.03 * np.random.normal(0, 1, next_state.shape)
        next_state = next_state / np.linalg.norm(next_state)

        reward = reward_function(next_state, prop, self.stp)
        fidelity = state_fidelity(next_state)
        doned = False
        
        if fidelity > self.maxfid:
            self.maxfid = state_fidelity(next_state)
        if fidelity > 0.95:
            doned = True

        next_states = [next_state[i, 0]
                       for i in range(chain_length)]  # 'matrix' to list
        next_states = np.array(
            list(itertools.chain(*[(i.real, i.imag) for i in next_states]))
        )  # complex to real vector

        self.state = next_states  # this vector is input to the network
        return next_states, reward, doned, fidelity


    
