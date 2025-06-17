
from state_env import State
import pandas as pd
import numpy as np
import configparser
import matplotlib.pyplot as plt
import sys

chain_length = sys.argv[1]

noisy_env = State()  # noisy environment
ideal_env = State()  # ideal environment

number_of_episodes = 100

noise_effects = pd.DataFrame(columns=['episode','noise_amplitude', 'noise_probability', 'max_fidelity'])    # we store the mean fidelity for each noise amplitude and probability
noise_details = pd.DataFrame(columns=['episode','time_step','noise_amplitude', 'noise_probability', 'fid_value','inner_product'])


for noise_amplitude in np.linspace(0, 1, 21):
    for noise_probability in np.linspace(0, 1, 21):

        print(f"Testing with noise_amplitude: {noise_amplitude}, noise_probability: {noise_probability}")
        lth = 5*chain_length

        actionspace = []  # store successful actions
        Qvalue = []  # total reward
        fid_max_vector = []  # max. fidelity in each episode
        t_fid_max_vector = []  # time of max. fidelity
        fid_end_vector = []  # final fidelity
        t_end_vector = []  # time of final fidelity
        success_action_sequences = []  # store successful success_action_seq

        for episode in range(number_of_episodes):
            # Generate a complex normalized vector of 16 components
            noisy_observation = noisy_env.reset()
            ideal_observation = ideal_env.reset()
            
            evolution = []
            Q = 0
            fid_max = 0
            t_fid_max = 0

            for i in range(lth):  # episode maximum length


                noisy_observation_, reward, done, fidelity = noisy_env.noisy_step(0,noise_amplitude=noise_amplitude,noise_probability=noise_probability)  # take action in the noisy_environment
                ideal_observation_, reward_ideal, done_ideal, fidelity_ideal = ideal_env.step(0)  # take action in the ideal_environment

                
                evolution.append(fidelity)  # store fidelity values
                noisy_observation = noisy_observation_  # Update current state
                ideal_observation = ideal_observation_  # Update current state

                inner_product = np.vdot(noisy_observation, ideal_observation)


                noise_details = pd.concat(
                    [
                        noise_details,
                        pd.DataFrame(
                            {
                                "episode": [episode],
                                "time_step": [i],
                                "noise_amplitude": [noise_amplitude],
                                "noise_probability": [noise_probability],
                                "fid_value": [fidelity],
                                "inner_product": [inner_product],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
            noise_effects = pd.concat(
                [
                    noise_effects,
                    pd.DataFrame(
                        {
                            "episode": [episode],
                            "noise_amplitude": [noise_amplitude],
                            "noise_probability": [noise_probability],
                            "max_fidelity": [max(evolution)],
                        }
                    ),
                ],
                ignore_index=True,
            )

noise_details.to_csv(f"n{chain_length}_noise_details.csv", index=False)
noise_effects.to_csv(f"n{chain_length}_noise_effects.csv", index=False)

