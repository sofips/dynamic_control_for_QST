from state_env import State
import pandas as pd
import numpy as np
import configparser
import matplotlib.pyplot as plt
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

chain_length = int(sys.argv[1])

number_of_episodes = 20

noise_effects = []
noise_details = []

config = configparser.ConfigParser()
config.read("config.ini")  # read the configuration file

def run_noise_case(noise_amplitude, noise_probability):
    noisy_env = State()  # noisy environment
    ideal_env = State()  # ideal environment

    lth = 5 * chain_length
    local_noise_effects = []
    local_noise_details = []

    for episode in range(number_of_episodes):
        noisy_observation = noisy_env.reset()
        ideal_observation = ideal_env.reset()
        evolution = []

        for i in range(lth):
            noisy_observation_, reward, done, fidelity = noisy_env.noisy_step(
                0, noise_amplitude=noise_amplitude, noise_probability=noise_probability
            )
            ideal_observation_, reward_ideal, done_ideal, fidelity_ideal = ideal_env.step(0)

            evolution.append(fidelity)
            noisy_observation = noisy_observation_
            ideal_observation = ideal_observation_

            inner_product = np.vdot(noisy_observation, ideal_observation)

            local_noise_details.append({
                "episode": episode,
                "time_step": i,
                "noise_amplitude": noise_amplitude,
                "noise_probability": noise_probability,
                "fid_value": fidelity,
                "inner_product": inner_product,
            })

        local_noise_effects.append({
            "episode": episode,
            "noise_amplitude": noise_amplitude,
            "noise_probability": noise_probability,
            "max_fidelity": max(evolution),
        })

    return local_noise_effects, local_noise_details

params = [
    (noise_amplitude, noise_probability)
    for noise_amplitude in np.linspace(0, 1, 21)
    for noise_probability in np.linspace(0, 1, 21)
]

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(run_noise_case, na, np_) for na, np_ in params]
    for future in as_completed(futures):
        local_noise_effects, local_noise_details = future.result()
        noise_effects.extend(local_noise_effects)
        noise_details.extend(local_noise_details)

noise_effects_df = pd.DataFrame(noise_effects)
noise_details_df = pd.DataFrame(noise_details)

noise_details_df.to_csv(f"n{chain_length}_noise_details.csv", index=False)
noise_effects_df.to_csv(f"n{chain_length}_noise_effects.csv", index=False)
