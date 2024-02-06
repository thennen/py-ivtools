import pickle
import concurrent.futures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from uncertainties import ufloat
from index import print_execution_time, index

"""
function to plot based on the extracted data
"""

def plot_measurement(data, figure_path=None, show=False):

    if isinstance(data, str):
        with open(data, 'rb') as f:
            # Load the pickled dictionary
            data = pickle.load(f) 

    if not isinstance(data, dict):
        raise Exception(
            "in plot_measurement(data): data must be instance of dict"
            f" but got {type(data)=}, {data=}")

    # extract variables
    filename = data["filename"]
    sample = data["samplename"]
    attenuation = data["attenuation"]
    pulse_width = data["pulse_width"]
    pulse_voltage = data["pulse_voltage"]
    t = data["t"]
    G = data["G"]
    V = data["V"]
    I = data["I"]
    pulse_times = data["pulse_times"]
    timestamp = data["timestamp"]
    total_pulses = data["total_pulses"]
    nplc = data["nplc"]
    V_set = data["V_set"]
    V_reset = data["V_reset"]
    num_sweeps = data["num_sweeps"]
    HCS = np.array(data["HCS"])
    LCS = np.array(data["LCS"])
    HCS_err = np.array(data["HCS_err"])
    LCS_err = np.array(data["LCS_err"])
    old_HCS = np.array(data["old_HCS"])
    old_LCS = np.array(data["old_LCS"])
    mean_HCS = data["mean_HCS"]
    mean_LCS = data["mean_LCS"]
    initial_HRS = data["initial_HRS"]
    initial_LCS = data["initial_LCS"]
    initial_LCS_err = data["initial_LCS_err"]
    pulses_30 = data["pulses_30"]
    pulses_50 = data["pulses_50"]
    pulses_70 = data["pulses_70"]
    pulses_90 = data["pulses_90"]
    
    # make nice, well readable plot
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["figure.figsize"] = (10, 6)  # Adjust the figure size
    plt.rcParams['figure.facecolor'] = 'white'
    fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=False)  # Create a figure and axes
    title = (
        f"sample = {sample}, attenuation={attenuation}dB "
        f"(pulse voltage = {pulse_voltage:.2f}V), pulsewidth={pulse_width}s"
    )
    fig.suptitle(title)
    axs = axs.flatten()
    
    # cut off values at the beginning
    t = t[10:]
    I = I[10:]
    G = G[10:]

    # plot conductance and pulse times
    # resistance
    axs[0].plot(t, G, lw=1, marker='.', markersize=4, label="conductance")
    axs[0].vlines(pulse_times, ymin=np.min(G), ymax=np.max(G), color="yellow", lw=0.8, label="pulse times")
    axs[0].set_ylabel("conductance / μS")
    axs[0].set_xlabel("time / s")
    axs[0].set_xlim(0, 3)
    axs[0].grid(visible=True, which="both")
    axs[0].legend()
    
    # get a table
    table = axs[1].table(
        cellText=[
            [f"time = {timestamp}"],
            [f"pulses = {total_pulses}"],
            [f"nplc = {nplc}"],
            [f"V_set (sweep) = {V_set:.1f}"],
            [f"V_reset (sweep) = {V_reset:.1f}"],
            [f"num_sweeps = {num_sweeps}"],
            [f"mean_sweep_LCS = {mean_LCS:.0f} μS"],
            [f"mean_sweep_HCS = {mean_HCS:.0f} μS"],
            [f"initial_LCS = ({ufloat(initial_LCS, initial_LCS_err)}) μS"],
            [f"pulses to 30% = {pulses_30}"],
            [f"pulses to 50% = {pulses_50}"],
            [f"pulses to 70% = {pulses_70}"],
            [f"pulses to 90% = {pulses_90}"]
        ],
        colLabels=["PARAMETERS"],
        #colWidths=[0.2],
        loc='center',
        zorder=20
    )
    #table.set_fontsize(14)
    table.scale(1, 0.9)
    axs[1].axis("off")

    # now plot LRS and HRS evolution
    axs[2].errorbar(range(1, len(HCS)+1), HCS, yerr=HCS_err, label="HCS",
                    fmt='o', markersize=4, capsize=4, elinewidth=2, markeredgewidth=2)
    #axs[2].plot(range(1, len(old_HCS)+1), old_HCS, marker='x', markersize=5, lw=1)
    axs[2].errorbar(range(1, len(LCS)+1), LCS, yerr=LCS_err, label="LCS",
                   fmt='o', markersize=4, capsize=4, elinewidth=2, markeredgewidth=2)
    #axs[2].plot(range(1, len(old_LCS)+1), old_LCS, marker='x', markersize=5, lw=1)
    axs[2].grid(which='both', visible=True)
    axs[2].legend()
    axs[2].set_xlabel('sweep no')
    axs[2].set_ylabel('conductance / μS')
    
    # get raw measurement data
    axs[3].plot(t, I*1e3)
    axs[3].grid(which='both', visible=True)
    axs[3].set_xlabel('time / s')
    axs[3].set_ylabel('current / mA')
    
    # Adjust the layout to make space for the table
    # fig.subplots_adjust(right=0.75)  # Increase the right margin as needed
    fig.tight_layout()
    
    # Save the figure with the adjusted layout
    if not figure_path:
        figure_path = (
            f"{os.getcwd()}/plots/measurements/{filename[:-2]}.png"
        )
    plt.savefig(
        figure_path,
        bbox_inches='tight',
        dpi=150
    )
    if show:
        plt.show()
    else:
        plt.close()

    # save the matplotlib-data
    """
    data_path = (
        f"~/Documents/plots/matplotlib_data/plot_{timestamp}_sample={sample}_"
        f"pulsewidth={pulse_width}s_attenuation={attenuation}dB_Vset={V_set}.s"
    )
    data_path = os.path.expanduser(data_path)
    with open(data_path, 'wb') as f:
            # write to dictionary
            pickle.dump((fig,axs), f, protocol=pickle.HIGHEST_PROTOCOL)
    """

    return (fig,axs)

"""
calculate plots for all the measurements to save in measurements_extracted
"""

# make a wrapper function for process_file
def plot_measurement_wrapper (file_path):
    try:
        with open(file_path, 'rb') as f:
            # Load the pickled dictionary
            data = pickle.load(f)
        plot_measurement(data)
    except Exception as e:
        print(f"there was an error plotting the measurement: {e}, path={file_path}")


@print_execution_time
def plot_all():

    # get current measurement index
    try:
        idx = pd.read_csv("index.csv")
    except Exception as e:
        idx = index()

    # make short announcement
    print("plotting measurements:")

    # do not consider files with an error
    idx = idx[idx["error"].isnull()]

    # get file_paths, last_modified
    make_path = lambda row: (
        f"{os.getcwd()}/plots/measurements/{row['filename'][:-2]}.png"
    )
    idx["figure_path"] = idx.apply(make_path, axis=1)
    idx["figure_exists"] = idx["figure_path"].apply(os.path.isfile)
    idx["figure_modified"] = idx[idx["figure_exists"]==True].loc[:,"extracted_path"].apply(os.path.getmtime)

    # get the unprocessed files
    unprocessed = idx[idx["figure_modified"].isnull()].loc[:,"extracted_path"].to_numpy()
    
    # get the modified files
    modified = idx[idx["figure_modified"] < idx["last_modified"]].loc[:,"filepath"].to_numpy()

    # get the file paths to process and shuffle them
    file_paths = np.concatenate([unprocessed, modified])
    np.random.shuffle(file_paths)

    # Create a progress bar
    progress_bar = tqdm(total=len(file_paths))

    # Create a ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for file_path in file_paths:
            # Submit the process_file function to the executor
            future = executor.submit(plot_measurement_wrapper, file_path)
            # add the future object to the list
            futures.append(future)

        # Process the completed futures
        for future in concurrent.futures.as_completed(futures):
            # Update the progress bar
            progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()


@print_execution_time
def plot_select():

    # get current measurement index
    try:
        idx = pd.read_csv("index.csv")
    except Exception as e:
        idx = index()

    # make short announcement
    print("plotting measurements:")

    # do not consider files with an error
    idx = idx[idx["error"].isnull()]

    # select figures you want
    idx = idx[idx["pulse_width"]==5e-11]
    idx = idx[idx["attenuation"]==6]
    idx = idx[idx["V_set"]==1.5]
    idx = idx.head()

    # get measurement data paths
    paths = idx["extracted_path"].to_numpy()
    filenames = idx["filename"].to_numpy()

    # plot measurements
    for path, filename in zip(paths, filenames):
        plot_measurement(
            path, 
            figure_path=f"{os.getcwd()}/plots/measurements/{filename[:-2]}.png"
        )


if __name__ == "__main__":
    plot_all()
    # plot_select()