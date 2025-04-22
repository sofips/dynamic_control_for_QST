import numpy as np
import os
import pandas as pd
import configparser

# plotting specifications
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from cycler import cycler
from tabulate import tabulate
from scipy.linalg import expm
import scipy.linalg as la
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D

mpl.rcParams.update({"font.size": 14})
plt.rcParams["axes.axisbelow"] = True
mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["lines.linewidth"] = 2
color_names = [
    "blue",
    "red",
    "green",
    "black",
    "magenta",
    "y",
    "darkorange",
]
mpl.rcParams["axes.prop_cycle"] = cycler(color=color_names)
pd.set_option("display.max_columns", None)
from nltk import ngrams
from collections import Counter


def uniformize_data(used_algorithm, **kwargs):

    if used_algorithm == "ga":

        directory = kwargs.get("directory", None)
        n = kwargs.get("n", None)

        files = os.listdir(directory)
        ga_actions = np.empty([0, n * 5])
        number_of_sequences = 0

        for file in files:
            if "act_sequence" in file:
                actions = np.genfromtxt(directory + file, dtype=int)
                ga_actions = np.vstack([ga_actions, actions])

        return ga_actions

    if used_algorithm == "zhang":
        file = kwargs.get("file", None)
        drl_actions = np.genfromtxt(file, dtype=int)
        return drl_actions

    if used_algorithm == "sp":
        file = kwargs.get("file", None)
        drl_actions = []

        with open(file, "r") as f:
            for line in f:
                sequence = line.split()
                sequence.pop()
                sequence = [int(x) for x in sequence]
                drl_actions.append(sequence)

            max_length = max(len(seq) for seq in drl_actions)
            padded_actions = np.array(
                [
                    np.pad(
                        seq, (0, max_length - len(seq)), "constant", constant_values=-1
                    )
                    for seq in drl_actions
                ]
            )

        return padded_actions


def ngram(sequences, title, n, max_ngrams=50, hex_code="#DDFFDD"):
    """
    Generate and plot a histogram of n-gram frequencies from a given set of pulse sequences.

    Parameters:
    sequences (np.ndarray): A 2D numpy array where each row represents a sequence of actions.
    title (str): The title of the plot.
    n (int): The length of the n-grams to generate.
    max_ngrams (int, optional): The maximum number of most common n-grams to display in the histogram. Default is 50.
    hex_code (str, optional): The hex color code for the bars in the histogram. Default is "#DDFFDD".

    Returns:
    None
    """

    n_sequences = np.shape(sequences)[0]

    all_ngrams = []

    for i in range(np.shape(sequences)[0]):
        for j in range(np.shape(sequences)[1]):
            sequences[i, j] = int(sequences[i, j])

    for i in range(n_sequences):
        # Generate n-grams for each sequence
        sequence_ngrams = list(ngrams(sequences[i], n))
        sequence_ngrams = [tuple(int(x) for x in ngram) for ngram in sequence_ngrams]

        # Filter out n-grams that contain a negative value
        filtered_ngrams = [
            ngram for ngram in sequence_ngrams if all(x >= 0 for x in ngram)
        ]
        all_ngrams.extend(filtered_ngrams)

    # Count n-gram frequencies
    ngram_counts = Counter(all_ngrams)
    total_count = sum(ngram_counts.values())

    # Get the "max_ngrams" most common n-grams
    most_common_ngrams = ngram_counts.most_common(max_ngrams)

    # Extract n-grams and their counts
    ngrams_list, counts = zip(*most_common_ngrams)

    # Plot histogram of n-gram frequencies
    figure, ax = plt.subplots(figsize=(12, 4))

    print(f"Total n-grams: {total_count}, Shown: {np.sum(counts)}")

    # Plot histogram
    ax.bar(range(len(ngrams_list)), counts, edgecolor="black", color=hex_code)

    # Configure x-ticks to show n-grams
    ax.set_xticks(range(len(ngrams_list)))
    ax.set_xticklabels(
        [f"{'-'.join(map(str, ngram))}" for ngram in ngrams_list], rotation=90
    )

    # Configure y-ticks to show percentage of total n-grams
    max_value = int(np.max(counts))
    y = np.linspace(0, max_value, 10, dtype=int)
    ax.set_yticks(y)
    y_ticks = np.around(y * 100 / total_count, 2)
    ax.set_yticklabels(y_ticks)

    # Set grid, title, and labels
    plt.grid()
    plt.title(f"{title}. Total of {n_sequences} sequences")
    ax.set_xlabel("{}-gram".format(n))
    ax.set_ylabel("%")
    plt.tight_layout()
    plt.show()


def plot_contour(
    sequences, title="Contour Plot", xlabel="Time Step", ylabel="Action Number"
):
    """
    Plots a contour plot for a given set of action sequences.

    Parameters:
    - sequences: 2D numpy array where rows represent action sequences and columns represent time steps.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    """
    n_sequences, max_time_step = sequences.shape
    action_counts = np.zeros((int(np.max(sequences)), max_time_step))

    for i in range(n_sequences):
        for time_step in range(max_time_step):
            action = sequences[i, time_step]
            for action_index in range(int(np.max(sequences))):
                if action == action_index:
                    action_counts[action_index, time_step] += 1

    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(range(max_time_step), range(action_counts.shape[0]))
    Z = action_counts

    contour = plt.contourf(X, Y, Z, cmap="viridis")
    plt.colorbar(contour)
    plt.grid(True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# calculation of state properties


def state_fidelity(state):
    """
    Calculate the fidelity of a given quantum state.

    The fidelity is computed as the real part of the product of the last element
    of the state vector and its complex conjugate.

    Parameters:
    state (numpy.ndarray): A complex numpy array representing the quantum state.

    Returns:
    float: The fidelity of the quantum state.
    """
    nh = np.shape(state)[0]
    fid = np.real(state[nh - 1] * np.conjugate(state[nh - 1]))
    return fid


def calc_exp_value(state, op):
    """
    Calculate the expected value of an operator given a quantum state.

    This function computes the expected value (or expectation value) of a given
    operator with respect to a provided quantum state. The expected value is
    calculated using the formula: ⟨state|op|state⟩.

    Parameters:
    state (numpy.ndarray): A complex vector representing the quantum state.
    op (numpy.ndarray): A complex matrix representing the operator.

    Returns:
    float: The real part of the expected value of the operator.
    """
    val = np.matmul(np.conjugate(np.transpose(state)), np.matmul(op, state))
    return np.real(val)


def calc_ipr(state):
    """
    Calculate the Inverse Participation Ratio (IPR) of a given state.

    The IPR is a measure of the localization of a state. It is calculated as the sum of the squared magnitudes of the state's components.

    Parameters:
    state (numpy.ndarray): A complex-valued numpy array representing the state vector.

    Returns:
    float: The calculated IPR value.
    """
    nh = np.shape(state)[0]
    inv_ipr = 0
    for i in range(nh):
        inv_ipr += np.real(state[i] * np.conjugate(state[i])) ** 2
    return 1 / inv_ipr


def calc_localization(state):
    """
    Calculate the localization of a given quantum state.

    The localization is computed as the sum of the squared magnitudes of the
    state's components, weighted by their position index.

    Parameters:
    state (numpy.ndarray): A 1D array representing the quantum state.

    Returns:
    float: The calculated localization value.
    """
    nh = np.shape(state)[0]
    loc = 0
    for i in range(nh):
        loc += np.real(state[i] * np.conjugate(state[i])) ** 2 * (i + 1)
    return loc


def action_selector(actions_name, b, nh):

    if actions_name == "original":
        actions = actions_zhang(b, nh)
    elif actions_name == "zhang":
        actions = actions_zhang(b, nh)
    elif actions_name == "oaps":
        actions = one_field_actions(b, nh)
    elif actions_name == "extra":
        actions = one_field_actions_extra(b, nh)
    else:
        print("Taking action matrix as input ...")
        actions = actions_name

    return actions


# plots for action sequences


def plot_single_sequence(
    action_sequence, nh, dt=0.15, b=100, label="", actions="original", add_natural=False
):

    action_sequence = [int(x) for x in action_sequence]
    t_steps = len(action_sequence) + 1

    # generar propagadores
    actions = action_selector(actions, b, nh)
    propagators = gen_props(actions, nh, dt)
    times = np.arange(0, t_steps, 1)

    # definicion del estado inicial e inicializacion de estados forzado y natural

    initial_state = np.zeros(nh, dtype=np.complex_)
    initial_state[0] = 1.0

    free_state = initial_state

    if add_natural:
        natural_evol = [state_fidelity(free_state)]

        nat_sequence = np.zeros(int(t_steps - 1), dtype=int)

        for action in nat_sequence:

            free_state = calculate_next_state(free_state, 0, propagators)
            natural_evol.append(state_fidelity(free_state))

        max_natural = np.max(natural_evol)

        plt.plot(
            times,
            natural_evol,
            "-v",
            label="Natural Evolution , Máx: {}".format(max_natural),
        )

    # inicializacion de estado forzado
    forced_state = initial_state

    # almacenar evolucion natural y evolucion forzada
    forced_evol = [state_fidelity(forced_state)]

    for action in action_sequence:

        forced_state = calculate_next_state(forced_state, action, propagators)
        forced_evol.append(state_fidelity(forced_state))

    max_forced = np.max(forced_evol)
    max_time = np.argmax(forced_evol)

    plt.plot(
        times,
        forced_evol,
        "-o",
        label=label + ". Máx.: {:.5f} at step {}".format(max_forced, max_time),
    )

    plt.legend(loc="upper left")
    # plt.show()


def plot_exp_value(
    action_sequence, nh, dt=0.15, b=100, label="", actions="original", add_actions=False
):

    if add_actions:
        add_actions = plt.subplots()

    action_sequence = [int(x) for x in action_sequence]
    t_steps = len(action_sequence)

    # generar propagadores
    actions = action_selector(actions, b, nh)
    propagators = gen_props(actions, nh, dt)
    times = np.arange(0, t_steps, 1)

    # definicion del estado inicial e inicializacion de estados forzado y natural

    initial_state = np.zeros(nh, dtype=np.complex_)
    initial_state[0] = 1.0

    zero_action = actions_zhang(b, nh)[0]

    # inicializacion de estado forzado
    forced_state = initial_state

    # almacenar evolucion natural y evolucion forzada
    exp_values = []

    for action in action_sequence:

        forced_state = calculate_next_state(forced_state, action, propagators)
        exp_values.append(calc_exp_value(forced_state, actions[action]))

    max_exp_val = np.max(exp_values)

    plt.plot(times, exp_values, "-o", label=label + ". Máx.: {}".format(max_exp_val))

    plt.legend(loc="upper left")

    if add_actions:
        ax2 = add_actions[1].twinx()

        color = "tab:grey"
        ax2.plot(action_sequence, "--o", label="Acciones", color=color)

        ax2.tick_params(axis="y", labelcolor=color)

        add_actions[0].tight_layout()


def plot_ipr(
    action_sequence, nh, dt=0.15, b=100, label="", actions="original", add_actions=False
):

    if add_actions:
        add_actions = plt.subplots()

    action_sequence = [int(x) for x in action_sequence]
    t_steps = len(action_sequence) + 1

    # generar propagadores
    actions = action_selector(actions, b, nh)
    propagators = gen_props(actions, nh, dt)
    times = np.arange(0, t_steps, 1)

    # definicion del estado inicial e inicializacion de estados forzado y natural

    initial_state = np.zeros(nh, dtype=np.complex_)
    initial_state[0] = 1.0

    zero_action = actions_zhang(b, nh)[0]

    # inicializacion de estado forzado
    forced_state = initial_state

    # almacenar evolucion natural y evolucion forzada
    ipr_values = [calc_ipr(forced_state)]

    for action in action_sequence:

        forced_state = calculate_next_state(forced_state, action, propagators)
        ipr_values.append(calc_ipr(forced_state))

    max_exp_val = np.max(ipr_values)

    plt.plot(times, ipr_values, "-o", label=label + ". Máx.: {}".format(max_exp_val))

    plt.legend(loc="upper left")

    if add_actions:
        ax2 = add_actions[1].twinx()

        color = "tab:grey"
        ax2.plot(action_sequence, "--o", label="Acciones", color=color)

        ax2.tick_params(axis="y", labelcolor=color)

        add_actions[0].tight_layout()


def plot_localization(
    action_sequence, nh, dt=0.15, b=100, label="", actions="original", add_actions=False
):

    if add_actions:
        add_actions = plt.subplots()

    action_sequence = [int(x) for x in action_sequence]
    t_steps = len(action_sequence) + 1

    # generar propagadores
    actions = action_selector(actions, b, nh)
    propagators = gen_props(actions, nh, dt)
    times = np.arange(0, t_steps, 1)

    # definicion del estado inicial e inicializacion de estados forzado y natural

    initial_state = np.zeros(nh, dtype=np.complex_)
    initial_state[0] = 1.0

    zero_action = actions_zhang(b, nh)[0]

    # inicializacion de estado forzado
    forced_state = initial_state

    # almacenar evolucion natural y evolucion forzada
    loc_values = [calc_localization(forced_state)]

    for action in action_sequence:

        forced_state = calculate_next_state(forced_state, action, propagators)
        loc_values.append(calc_localization(forced_state))

    max_loc_values = np.max(loc_values)

    plt.plot(times, loc_values, "-o", label=label + ". Máx.: {}".format(max_loc_values))

    plt.legend(loc="upper left")

    if add_actions:
        ax2 = add_actions[1].twinx()

        color = "tab:grey"
        ax2.plot(action_sequence, "--o", label="Acciones", color=color)

        ax2.tick_params(axis="y", labelcolor=color)

        add_actions[0].tight_layout()


def plot_all_metrics(
    action_sequence,
    nh,
    dt=0.15,
    b=100,
    label="",
    actions="original",
    add_natural=False,
    add_actions=False,
):
    """
    Genera una grilla 2x2 con las siguientes gráficas:
    - Fidelidad (single_sequence)
    - IPR
    - Valor esperado (exp_value)
    - Localización
    """
    action_sequence = [int(x) for x in action_sequence]
    t_steps = len(action_sequence) + 1
    times = np.arange(0, t_steps, 1)

    # Generar propagadores
    actions = action_selector(actions, b, nh)
    propagators = gen_props(actions, nh, dt)

    # Inicializar estado inicial
    initial_state = np.zeros(nh, dtype=np.complex_)
    initial_state[0] = 1.0

    # Configurar la figura
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    axs = axs.ravel()  # Facilita el acceso a los subplots

    # Función auxiliar para agregar acciones
    def plot_actions(ax):
        if add_actions:
            ax2 = ax.twinx()
            color = "tab:grey"
            ax2.plot(
                range(len(action_sequence)),
                action_sequence,
                "--o",
                label="Acciones",
                color=color,
            )
            ax2.tick_params(axis="y", labelcolor=color)
            ax2.legend(loc="upper right")

    # 1. Fidelidad
    free_state = initial_state
    forced_state = initial_state
    forced_evol = [state_fidelity(forced_state)]

    for action in action_sequence:
        forced_state = calculate_next_state(forced_state, action, propagators)
        forced_evol.append(state_fidelity(forced_state))
    axs[0].plot(times, forced_evol, "-o", label=f"{label}. Máx.: {max(forced_evol)}")
    axs[0].set_title("Fidelidad")
    axs[0].legend(loc="upper left")

    if 0 in add_actions:
        plot_actions(axs[0])

    # 2. IPR
    forced_state = initial_state
    ipr_values = [calc_ipr(forced_state)]

    for action in action_sequence:
        forced_state = calculate_next_state(forced_state, action, propagators)
        ipr_values.append(calc_ipr(forced_state))
    axs[1].plot(times, ipr_values, "-o", label=f"{label}. Máx.: {max(ipr_values)}")
    axs[1].set_title("IPR")
    axs[1].legend(loc="upper left")

    if 1 in add_actions:
        plot_actions(axs[1])

    # 3. Valor esperado
    forced_state = initial_state
    exp_values = []
    for action in action_sequence:
        forced_state = calculate_next_state(forced_state, action, propagators)
        exp_values.append(calc_exp_value(forced_state, actions[action]))
    axs[2].plot(times[:-1], exp_values, "-o", label=f"{label}. Máx.: {max(exp_values)}")
    axs[2].set_title("Valor Esperado")
    axs[2].legend(loc="upper left")
    if 2 in add_actions:
        plot_actions(axs[2])

    # 4. Localización
    forced_state = initial_state
    loc_values = [calc_localization(forced_state)]
    for action in action_sequence:
        forced_state = calculate_next_state(forced_state, action, propagators)
        loc_values.append(calc_localization(forced_state))
    axs[3].plot(times, loc_values, "-o", label=f"{label}. Máx.: {max(loc_values)}")
    axs[3].set_title("Localización")
    axs[3].legend(loc="upper left")

    if 3 in add_actions:
        plot_actions(axs[3])

    if add_natural:
        natural_evol = [state_fidelity(free_state)]

        # generar propagadores
        actions = action_selector("original", b, nh)
        propagators = gen_props(actions, nh, dt)

        for action in action_sequence:
            free_state = calculate_next_state(free_state, 0, propagators)
            natural_evol.append(state_fidelity(free_state))
        axs[0].plot(times, natural_evol, "-v", label="Evolución natural")
        axs[0].legend(loc="upper left")
    # Ajustar diseño
    plt.tight_layout()
    plt.show()


def find_max(action_sequences, nh, b=100, dt=0.15, actions="original"):

    max_fid = 0.0
    max_index = 0.0

    # generar propagadores
    actions = action_selector(actions, b, nh)
    propagators = gen_props(actions, nh, dt)

    for i in range(np.shape(action_sequences)[0]):
        action_sequence = action_sequences[i][:]
        action_sequence = [int(x) for x in action_sequence]

        t_steps = len(action_sequence) + 1
        times = np.arange(0, t_steps, 1)

        # definicion del estado inicial e inicializacion de estados forzado y natural

        initial_state = np.zeros(nh, dtype=np.complex_)
        initial_state[0] = 1.0

        # inicializacion de estado forzado
        forced_state = initial_state

        # almacenar evolucion natural y evolucion forzada
        forced_evol = [state_fidelity(forced_state)]

        for action in action_sequence:

            forced_state = calculate_next_state(forced_state, action, propagators)
            forced_evol.append(state_fidelity(forced_state))

            max_forced = np.max(forced_evol)

        if max_forced > max_fid:
            max_fid = max_forced
            max_index = i
    print("Max fid:", max_fid, "Max Index:", max_index)
    return max_fid, max_index


def one_field_actions(bmax, nh):

    action_matrices = np.zeros((nh + 1, nh, nh))
    J = 1.0

    for i in range(0, nh):

        for k in range(0, nh - 1):
            action_matrices[i + 1, k, k + 1] = J
            action_matrices[i + 1, k + 1, k] = action_matrices[i + 1, k, k + 1]

        action_matrices[i + 1, i, i] = bmax

    for k in range(0, nh - 1):
        action_matrices[0, k, k + 1] = J
        action_matrices[0, k + 1, k] = action_matrices[0, k, k + 1]

    return action_matrices


def diagonals_zhang(bmax, i, nh):
    """
    Construction of diagonals associated to referenced work. The first and last three sites
    can be controlled.

    Parameters:
    bmax (float): Control field value.
    i (int): The index determining which diagonal elements to set to 1.
    nh (int): The length of the spin chain, corresponding to the action
    matrices size.

    Returns:
    numpy.ndarray: A diagonal vector of length `nh` with specific elements set to `bmax` based on the index `i`,
    corresponding to the 16 action matrices.
    """

    b = np.full(nh, 0)

    if i == 1:
        b[0] = 1

    elif i == 2:

        b[1] = 1

    elif i == 3:

        b[0] = 1
        b[1] = 1

    elif i == 4:
        b[2] = 1  # correccion

    elif i == 5:
        b[0] = 1
        b[2] = 1

    elif i == 6:
        b[1] = 1
        b[2] = 1

    elif i == 7:
        b[0] = 1
        b[1] = 1
        b[2] = 1

    elif i == 8:
        b[nh - 3] = 1

    elif i == 9:
        b[nh - 2] = 1

    elif i == 10:
        b[nh - 3] = 1
        b[nh - 2] = 1

    elif i == 11:
        b[nh - 1] = 1

    elif i == 12:
        b[nh - 3] = 1
        b[nh - 1] = 1

    elif i == 13:
        b[nh - 2] = 1
        b[nh - 1] = 1

    elif i == 14:
        b[nh - 3] = 1
        b[nh - 2] = 1
        b[nh - 1] = 1

    elif i == 15:
        b[:] = 1
    else:
        b = np.full(nh, 0.0)  # correccion

    b = bmax * b

    return b


def actions_zhang(bmax, nh):

    actions = np.zeros((16, nh, nh))

    for i in range(0, 16):

        b = diagonals_zhang(bmax, i, nh)

        J = 1

        for k in range(0, nh - 1):
            actions[i, k, k + 1] = J
            actions[i, k + 1, k] = actions[i, k, k + 1]

        for k in range(0, nh):

            actions[i, k, k] = b[k]

    return actions


def fid_evolution(
    action_sequence, nh, dt=0.15, b=100, label="", actions="original", add_natural=False
):

    action_sequence = [int(x) for x in action_sequence]
    t_steps = len(action_sequence) + 1

    # generar propagadores
    actions = action_selector(actions, b, nh)
    propagators = gen_props(actions, nh, dt)

    # inicializacion de estados
    initial_state = np.zeros(nh, dtype=np.complex_)
    initial_state[0] = 1.0

    forced_state = initial_state
    free_state = initial_state

    # almacenar evolucion natural y evolucion forzada
    forced_evol = [state_fidelity(forced_state)]

    for action in action_sequence:

        forced_state = calculate_next_state(forced_state, action, propagators)
        forced_evol.append(state_fidelity(forced_state))

    if add_natural:
        natural_evol = [state_fidelity(free_state)]

        nat_sequence = np.zeros(int(t_steps - 1), dtype=int)

        for action in nat_sequence:

            free_state = calculate_next_state(free_state, 0, propagators)
            natural_evol.append(state_fidelity(free_state))

        return forced_evol, natural_evol

    else:

        return forced_evol


def gen_props(actions, n, dt, test=True):
    """
    Generate propagators for a set of action matrices.

    Parameters:
    actions (numpy.ndarray): A 3D array of shape (n_actions, n, n) containing the action matrices.
    n (int): The dimension of the action matrices equal to the chain length.
    dt (float): The time step for propagation.
    test (bool, optional): If True, perform a test to check the correctness of the propagation. Default is True.

    Returns:
    numpy.ndarray: A 3D array of shape (n_actions, n, n) containing the propagators.

    Notes:
    - The function first diagonalizes each action matrix to obtain eigenvalues and eigenvectors.
    - It then constructs the propagators using the matrix exponential.
    - If `test` is True, it checks if the eigenstates are properly propagated and prints the result.
    """

    n_actions = actions.shape[0]
    comp_i = complex(0, 1)
    props = np.zeros((n_actions, n, n), dtype=np.complex_)

    for i in range(0, n_actions):  # propagator building
        props[i, :, :] = expm(-1j * actions[i] * dt)

    if test:

        en = np.zeros((n_actions, n), dtype=np.complex_)
        bases = np.zeros((n_actions, n, n), dtype=np.complex_)

        for j in range(0, n_actions):  # diagonalization of action matrices
            en[j, :], bases[j, :, :] = la.eig(actions[j, :, :])

        for a in np.arange(0, n_actions):
            for j in np.arange(0, n):
                errores = (
                    calculate_next_state(
                        bases[a, :, j], a, props, check_normalization=True
                    )
                    - np.exp(-comp_i * dt * en[a, j]) * bases[a, :, j]
                )
                et = np.sum(errores)
                if la.norm(et) > 1e-8:
                    raise ValueError(
                        (
                            "Propagation Error: Eigenstates are not being "
                            "properly propagated"
                        )
                    )

    return props


def fidelity(action_sequence, props, return_time=False, test_normalization=True):
    """
    Calculate the fidelity resulting of a given pulse sequence. The state is
    initialized to /10...0>

    Parameters:
    action_sequence (list or array-like): A sequence of actions to be applied
    to the initial state.
    props (ndarray): A 3D array where props[action] is the propagation matrix
    corresponding to that action.
    return_time (bool, optional): If True, return the time step at which the
    maximum fidelity is achieved. Default is False.
    test_normalization (bool, optional): If True, test the normalization of
    the final state. Default is True.

    Returns:
    float: The maximum fidelity achieved.
    tuple: If return_time is True, returns a tuple (max_fid, imax) where
    max_fid is the maximum fidelity and imax is the time step at which it is achieved.
    """

    n = np.shape(props)[1]
    state = np.zeros(n, dtype=np.complex_)
    state[0] = 1.0
    max_fid = 0.0
    imax = 0
    i = 0

    for action in action_sequence:

        i += 1
        state = np.matmul(props[action, :, :], state)
        fid = np.real(state[n - 1] * np.conjugate(state[n - 1]))

        if fid > max_fid:
            imax = i
            max_fid = fid

    if test_normalization:

        if abs(la.norm(state) - 1.0) > 1e-8:
            print("Normalization test failed. Norm of final state: ", la.norm(state))
            quit()

    if return_time:
        return max_fid, imax

    return max_fid


def calculate_next_state(state, action_index, props, check_normalization=True):
    """
    Calculate the next state by applying the propagator associated to an action.

    Args:
        state (np.ndarray): The current state represented as a numpy array.
        action_index (int): The index of the action to be applied.
        props (list or np.ndarray): The propagator corresponding to that action.

    Returns:
        np.ndarray: The next state after applying the action.

    Raises:
        SystemExit: If the normalization of the next state fails.
    """

    state = np.transpose(np.mat(state))
    p = props[action_index]
    next_state = p * state
    next_state = np.asarray(np.transpose(next_state))
    next_state = np.squeeze(next_state)

    if check_normalization:
        if abs(la.norm(next_state) - 1.0) > 1e-8:
            print("Normalization failed. Norm of state: ", la.norm(state))
            quit()

    return next_state


def plot_ga_solutions(
    directories,
    n,
    add_natural=False,
    fs=14,
):

    different_parameters = get_different_parameters(directories, print_params=False)
    directories = different_parameters.loc["directory"].values.tolist()

    labels = [
        ", ".join(
            f"{param}: {value}"
            for param, value in zip(
                different_parameters.index, different_parameters[col]
            )
            if param != "directory"
        )
        for col in different_parameters.columns
    ]

    legend_elements = []
    color_index = 0

    for directory, label in zip(directories, labels):
        directory = f"genetic_algorithm_results/{directory}"
        ga_sequences = uniformize_data(
            "ga", **{"directory": f"{directory}/n{n}/", "n": n}
        )

        config_files = [f for f in os.listdir(directory) if f.endswith(".ini")]
        if len(config_files) != 1:
            raise Exception("There must be exactly one .ini file in the directory.")

        # Load the configuration file
        config = configparser.ConfigParser()
        config.read(f"{directory}/{config_files[0]}")

        dt = config.getfloat("system_parameters", "dt")
        b = config.getfloat("system_parameters", "b")
        action_set = config.get("ga_initialization", "action_set")

        samples = np.arange(0, np.shape(ga_sequences)[0], 1)

        colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        color_index += 1

        for sample in samples:
            forced_evol, natural_evol = fid_evolution(
                ga_sequences[sample][:],
                n,
                dt=dt,
                b=b,
                actions=action_set,
                add_natural=True,
            )
            color = colors[color_index % len(colors)]
            plt.plot(
                forced_evol,
                "-o",
                color=color,
                alpha=0.5,
                linewidth=0.9,
                markersize=0.2,
            )

        legend_elements = legend_elements + [
            Line2D([0], [0], color=color, lw=1.2, label=label)
        ]

    if add_natural:
        plt.plot(
            natural_evol,
            "-o",
            label="Natural",
            color="slategrey",
            linewidth=5,
            markersize=0.2,
            zorder=-2,
        )
        legend_elements.append(
            Line2D([0], [0], color="slategrey", lw=3, label="Natural")
        )
    plt.xlabel("Time Step", fontsize=fs)
    plt.ylabel("Transition probability", fontsize=fs)
    plt.tight_layout()
    plt.legend(
        handles=legend_elements, fontsize=fs, loc="lower left", bbox_to_anchor=(0, 0.1)
    )


def plot_metric(directories, column, personalized_colors=False):

    index = 0
    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

    if personalized_colors:
        colors = personalized_colors

    different_parameters = get_different_parameters(directories, print_params=False)

    legend_elements = []
    directories = different_parameters.loc["directory"].values.tolist()
    print(directories)

    labels = [
        ", ".join(
            f"{param}: {value}"
            for param, value in zip(
                different_parameters.index, different_parameters[col]
            )
            if param != "directory"
        )
        for col in different_parameters.columns
    ]

    for directory, label in zip(directories, labels):

        file_path = os.path.join(
            f"genetic_algorithm_results/{directory}/", "nvsmaxfid.dat"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The file 'nvsmaxfid.dat' does not exist in the directory {directory}."
            )

        results_data = pd.read_csv(file_path)
        fs = 14

        grouped_df = results_data.groupby("n")
        mean = grouped_df[column].mean()
        std = grouped_df[column].std()
        min_value = grouped_df[column].min()
        max_value = grouped_df[column].max()

        dimensions = results_data["n"].unique()
        color = colors[index % len(colors)]

        # Plot mean with error bars
        plt.errorbar(dimensions, mean, yerr=std, fmt="o--", color=color, capsize=5)
        # Plot max values
        plt.plot(dimensions, max_value, "s-", color=color, linewidth=0.5)

        # Add legend entry for this directory
        legend_elements.append(
            Line2D([0], [0], color=color, linestyle="-", label=f"{label}")
        )

        index += 1

    # Add legend to the plot

    # Add legend entry for this directory
    legend_elements.append(
        Line2D(
            [0], [0], color="grey", marker="o", linestyle="--", label=f"Mean {column}"
        )
    )
    legend_elements.append(
        Line2D([0], [0], color="grey", marker="s", linestyle="-", label=f"Max {column}")
    )

    plt.legend(
        handles=legend_elements,
        fontsize=fs - 2,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
    )
    index += 1


def access_ga_params(directory, print_params=True):
    """
    Access and retrieve the GA parameters from a configuration file in the specified directory.

    Parameters:
    - directory (str): The directory containing the GA parameters file.
    - print_params (bool): If True, prints the parameters in a fancy grid format.
    Returns:
    - pd.DataFrame: A DataFrame containing the parameters with columns 'Section', 'Parameter', and 'Value'.

    Raises:
    - Exception: If there is not exactly one .ini file in the directory.
    """

    files = [f for f in os.listdir(directory) if f.endswith(".ini")]
    if len(files) != 1:
        raise Exception("There must be exactly one .ini file in the directory.")

    # Load the configuration file
    config = configparser.ConfigParser()
    config.read(f"{directory}/{files[0]}")

    # Convert to DataFrame
    data = []
    for section in config.sections():
        for key, value in config.items(section):
            data.append({"Section": section, "Parameter": key, "Value": value})

    df = pd.DataFrame(data)

    if print_params == True:
        # Print in a fancy grid
        print(f"Showing parameters for {directory}:")
        print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))

    return df


def get_parameter_value(df, parameter, print_value=False):
    row = df[(df["Parameter"] == parameter)]
    if not row.empty:
        value = row["Value"].values[0]
        if print_value:
            print(f"Parameter '{parameter}' :  {value}")
    else:
        raise ValueError(f"Parameter '{parameter}' not found.")

    return value


def get_different_parameters(directories, print_params=True):
    """
    Access and retrieve the GA parameters from configuration files in the
    specified directories,
    and compare across all parameters to identify those with different values, excluding 'speed_fraction' and 'directory'.

    Parameters:
    - directories (list): The directories containing the GA parameters files.
    - print_params (bool): If True, prints the parameters in a fancy grid format.

    Returns:
    - pd.DataFrame: A DataFrame showing parameters with different values across directories.
    """

    all_dfs = []
    for directory in directories:
        df = access_ga_params(directory, print_params=False)
        df["Directory"] = directory  # Add directory column for tracking
        all_dfs.append(df)

    # Concatenate all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Replace specific parameter names
    combined_df["Parameter"] = combined_df["Parameter"].replace(
        {"fitness_function": "fitness", "action_set": "actions"}
    )

    # Exclude 'speed_fraction' and 'directory' parameters
    combined_df = combined_df[~combined_df["Parameter"].isin(["speed_fraction"])]

    # Pivot the DataFrame to compare values across directories
    pivot_df = combined_df.pivot_table(
        index="Parameter", columns="Directory", values="Value", aggfunc="first"
    )

    # Identify parameters with different values
    different_values_df = pivot_df[pivot_df.nunique(axis=1) > 1]

    if print_params:
        print("\nParameters with Different Values:")
        print(tabulate(different_values_df, headers="keys", tablefmt="fancy_grid"))

    return different_values_df


def plot_max_fid_solutions(directories, n, add_natural=False, fs=14, nsamples=1):

    lines = [
        "--",
        "-o",
        "-v",
        "-s",
        "-*",
        "-^",
        "-<",
        "->",
        "-|",
        "-.",

    ]

    different_parameters = get_different_parameters(directories, print_params=False)
    directories = different_parameters.loc["directory"].values.tolist()

    labels = [
        ", ".join(
            f"{param}: {value}"
            for param, value in zip(
                different_parameters.index, different_parameters[col]
            )
            if param != "directory"
        )
        for col in different_parameters.columns
    ]

    legend_elements = []
    color_index = 0

    for directory, label in zip(directories, labels):
        directory = f"genetic_algorithm_results/{directory}"
        ga_sequences = uniformize_data(
            "ga", **{"directory": f"{directory}/n{n}/", "n": n}
        )

        config_files = [f for f in os.listdir(directory) if f.endswith(".ini")]
        if len(config_files) != 1:
            raise Exception("There must be exactly one .ini file in the directory.")

        # Load the configuration file
        config = configparser.ConfigParser()
        config.read(f"{directory}/{config_files[0]}")
        dt = config.getfloat("system_parameters", "dt")
        b = config.getfloat("system_parameters", "b")
        action_set = config.get("ga_initialization", "action_set")

        for sample in np.arange(0, nsamples, 1):

            max_fid, max_index = find_max(
                ga_sequences, n, b=b, dt=dt, actions=action_set
            )

            colors = color_names  # mpl.rcParams["axes.prop_cycle"].by_key()["color"]
            color_index += 1

            forced_evol, natural_evol = fid_evolution(
                ga_sequences[max_index][:],
                n,
                dt=dt,
                b=b,
                actions=action_set,
                add_natural=True,
            )
            color = colors[color_index % len(colors)]
            plt.plot(
                forced_evol,
                lines[color_index],
                color=color,
                alpha=2 / nsamples,
                linewidth=10 / nsamples/len(directories),
            )

            legend_elements = legend_elements + [
                Line2D(
                    [0],
                    [0],
                    linestyle=lines[color_index][0],
                    marker=lines[color_index][1],
                    color=color,
                    lw=1.2,
                    label=label + f". Max. fid = {max_fid:.4f}",
                )
            ]

            ga_sequences = np.delete(ga_sequences, max_index, axis=0)
    if add_natural:
        plt.plot(
            natural_evol,
            "--",
            label="Natural",
            color="slategrey",
            linewidth=5,
            zorder=-2,
            alpha = 0.77
        )
        legend_elements.append(
            Line2D([0], [0], color="slategrey", lw=3, label="Natural")
        )
    plt.xlabel("Time Step", fontsize=fs)
    plt.ylabel("Transition probability", fontsize=fs)
    plt.tight_layout()
    plt.legend(handles=legend_elements, fontsize=fs - 3, loc="upper left")
