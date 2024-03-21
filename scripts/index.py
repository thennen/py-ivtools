import os
import pickle
import hashlib
import time
import concurrent.futures
import pandas as pd
import numpy as np
from tqdm import tqdm
from uncertainties.unumpy import uarray, nominal_values, std_devs
import warnings

# Open a log file in write mode
log_file = open('index_warnings.log', 'w')

# Function to handle warnings and write to the log file
def warning_handler(message, category, filename, lineno, file=None, line=None):
    log_file.write(warnings.formatwarning(message, category, filename, lineno))

# Set the custom warning handler
warnings.showwarning = warning_handler

"""
decorator function to print execution time
"""

def print_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
        return result
    return wrapper

"""
hashing functions taken from stackoverflow, chatgpt
"""

def calculate_file_hash(file_path, algorithm="md5", chunk_size=8192):
    hash_algorithm = hashlib.new(algorithm)
    with open(file_path, 'rb') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            hash_algorithm.update(chunk)
    return hash_algorithm.hexdigest()

def hash_dir(dir_path):
    hashes = []
    for path, dirs, files in os.walk(dir_path):
        for file in sorted(files): # we sort to guarantee that files will always go in the same order
            hashes.append(calculate_file_hash(os.path.join(path, file)))
        for dir in sorted(dirs): # we sort to guarantee that dirs will always go in the same order
            hashes.append(hash_dir(os.path.join(path, dir)))
        break # we only need one iteration - to get files and dirs in current directory
    return str(hash(''.join(hashes)))

"""
process file function
"""

def process_file(file_path):

    # get filename from file_path
    file = file_path.split("/")[-1]
    
    # read file
    data = pd.read_pickle(file_path)

    # Extract the required information
    samplename = data["samplename"]
    padname = data["padname"]
    num_sweeps = len(data["resets"])
    attenuation = data["attenuation"]
    pulse_width = data["pulse_width"]
    points = data["points"]
    timestamp = data["timestamp"]
    V_set = np.round(np.max(data["sets"][0]["V"]),decimals=1)
    V_reset = np.round(np.min(data["resets"][0]["V"]),decimals=1)
    nplc = data["nplc"]
    def att_to_volt (attenuation):
        return 5*10**(-attenuation/20)
    pulse_voltage = att_to_volt(attenuation)
    last_modified = os.path.getmtime(file_path)
    error=""
    
    # analyse the sweeps
    # analyse the sweeps
    def get_conductance(data):
        V = np.abs(data["Vmeasured"])
        I = np.abs(data["I"])
        idx = np.where(
            (np.abs(V) >= 0.1) &
            (np.abs(V) <= 0.3)
        )[0]
        idx = idx[idx > len(V)//2]
        mean_resistance = np.mean(I[idx] / V[idx] * 1e6)
        std_resistance = np.std(I[idx] / V[idx] * 1e6, ddof=1)
        return mean_resistance, std_resistance
    LCS, LCS_err = zip(*[get_conductance(reset_data) for reset_data in data["resets"]])
    HCS, HCS_err = zip(*[get_conductance(set_data) for set_data in data["sets"]])

    def mean_min_max (arr: list):
        return (np.mean(arr), np.min(arr), np.max(arr))
    mean_LCS, min_LCS, max_LCS = mean_min_max(LCS)
    mean_HCS, min_HCS, max_HCS = mean_min_max(HCS)
    
    def old_get_conductance(data) -> np.double:
        V = data["Vmeasured"]
        I = data["I"]
        # find index of last voltage greater than or equal to 0.2V
        idx = np.where(
            (np.abs(V) >= 0.2)
        )[0][-1]
        return I[idx] / V[idx] * 1e6
    old_LCS = list(map(old_get_conductance, data["resets"]))
    old_HCS = list(map(old_get_conductance, data["sets"]))
    
    # extract Voltage (V), current(I), and time (t)
    # calculate conductance (G)
    V = data["Vmeasured"]
    I = data["I"]
    t = data["t"]
    G = I/V*1e6 # conductance in uS
    R = V/I*1e-3 # resistance in kilo ohms
    
    # drop nan values
    nan = np.sum(np.isnan(G))
    V = V[~np.isnan(G)]
    I = I[~np.isnan(G)]
    t = t[~np.isnan(G)]
    R = R[~np.isnan(G)]
    G = G[~np.isnan(G)]
    
    # calculate number of pulses
    # get the pulse times and convert from ns to s
    pulse_times = np.array(data["t_event"])[:-1] # cut off last value that is check pulse
    pulse_times = (pulse_times-data["t_begin"])/1e9
    total_pulses = len(pulse_times)

    # correct slight shift between python time and keithley time
    # 0.02s interval was found experimentally
    # mask those values that are out of bonds
    pulse_idx = np.searchsorted(t, pulse_times)
    left_idx = pulse_idx+20
    right_idx = left_idx-50
    valid_idx = (left_idx < len(G)) & (right_idx < len(G))
    left_idx = left_idx[valid_idx]
    right_idx = right_idx[valid_idx]
    left = np.searchsorted(t, 0.1)
    left_idx = np.insert(left_idx,0,left)
    right_idx = np.append(right_idx,len(G)-1)
    useful_idx = left_idx < right_idx
    left_idx = left_idx[useful_idx]
    right_idx = right_idx[useful_idx]
    
    # extract conductances
    # find out whether set was succesful
    def get_conductances(I, V, left_idx, right_idx):
        I_mean = np.array(list(map(
            lambda left, right: (
                np.mean(I[left:right+1])
            ), 
            left_idx, right_idx
        )))
        I_std = np.array(list(map(
            lambda left, right: (
                np.std(I[left:right+1], ddof=1)
            ), 
            left_idx, right_idx
        )))
        I = uarray(I_mean, I_std)
        V_mean = np.array(list(map(
            lambda left, right: (
                np.mean(V[left:right+1])
            ), 
            left_idx, right_idx
        )))
        V_std = np.array(list(map(
            lambda left, right: (
                np.std(V[left:right+1], ddof=1)
            ), 
            left_idx, right_idx
        )))
        V = uarray(V_mean, V_std)
        G = I / V * 1e6
        return nominal_values(G), std_devs(G)
        
    conductances, conductances_err = get_conductances(I, V, left_idx, right_idx)
    initial_LCS = conductances[0]
    initial_LCS_err = conductances_err[0]
    initial_HRS = 1e6/initial_LCS
    mean_conductance, min_conductance, max_conductance = mean_min_max(conductances)
    end_conductance = conductances[-1]
    initial_LCS_window = ""
    if initial_LCS < 100:
        initial_LCS_window = "initial LCS < 100 µS"
    else:
        initial_LCS_window = "initial LCS ≥ 100 µS"
    
    # find the switching number of pulses
    # number of pulses it took to
    # first reach 50%, 70%, and 90% 
    try: 
        def threshold_pulses(percent):
            threshold = percent * (max_conductance-min_conductance) + min_conductance
            return np.argwhere(conductances > threshold).min()
        pulses_30 = threshold_pulses(0.3)
        pulses_50 = threshold_pulses(0.5)
        pulses_70 = threshold_pulses(0.7)
        pulses_90 = threshold_pulses(0.9)
    except Exception as e:
        error += f"{e}\n"
        # print(f"file {file} causing error {e}")
        pulses_30 = np.NaN
        pulses_50 = np.NaN
        pulses_70 = np.NaN
        pulses_90 = np.NaN

    # analyse transient pulses
    t_tek_list = data["t_scope"], data["v_answer"]
    no_transient_measurements = len(t_tek_list)
    
    # define path of extracted measurement
    extracted_path = f"{os.getcwd()}/measurements_extracted/{file_path.split('/')[-1]}"

    # Create a dictionary of the extracted information
    data_dict = {
        "filename": file,
        "filepath": file_path,
        "extracted_path": extracted_path,
        "samplename": samplename,
        "padname": padname,
        "timestamp": timestamp,
        "last_modified": last_modified,
        "error": error,
        "nan_values": nan,
        "no_transient_measurements": no_transient_measurements,
        
        "nplc": nplc,
        "points": points,
        "num_sweeps": num_sweeps,
        "total_pulses": total_pulses,

        "V_set": V_set,
        "V_reset": V_reset,
        "attenuation": attenuation,
        "pulse_voltage": pulse_voltage,
        "pulse_width": pulse_width,
        "initial_HRS": initial_HRS,
        "initial_LCS": initial_LCS,
        "initial_LCS_err": initial_LCS_err,
        "initial_LCS_window": initial_LCS_window,
        
        "mean_conductance": mean_conductance,
        "end_conductance": end_conductance,
        "max_conductance": max_conductance,
        "min_conductance": min_conductance,
        
        "pulses_30": pulses_30,
        "pulses_50": pulses_50,
        "pulses_70": pulses_70,
        "pulses_90": pulses_90,
        
        "mean_HCS": mean_HCS,
        "max_HCS": max_HCS,
        "min_HCS": min_HCS,
        "mean_LCS": mean_LCS,
        "max_LCS": max_LCS,
        "min_LCS": min_LCS
    }

    # save results to dictionary
    results = {
        "V": V,
        "I": I,
        "t": t,
        "G": G,
        "R": R,
        "pulse_times": pulse_times,
        "conductances": conductances,
        "conductances_err": conductances_err,
        "HCS": HCS,
        "HCS_err": HCS_err,
        "old_HCS": old_HCS,
        "LCS": LCS,
        "LCS_err": LCS_err,
        "old_LCS": old_LCS
    }
    
    # add content of data dict
    results = results | data_dict

    # save extracted data to pickle
    with open(extracted_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Append the dictionary to the data list
    return data_dict
  
"""
Postprocess the created index
"""

def postprocess(df: pd.DataFrame) -> pd.DataFrame:

    # add a column that can be compared
    df['time'] = pd.to_datetime(df['timestamp'], format='%Y.%m.%d-%H.%M.%S')

    # Sort the DataFrame by 'timestamp' within each 'samplename' group
    df = df.sort_values(by=['samplename', 'timestamp'])

    # Calculate the cumulative count of each sample occurrence before the current row
    df['measurement_no'] = df.groupby('samplename').cumcount()

    return df

"""
make an index of all measurements. The index is a pandas dataframe containing important
parameters and the filepaths that belong with it. The dataframe is also saved to csv.

returns the index dataframe
"""

# make a wrapper function for process_file
def process_file_wrapper (file_path):
    try:
        return process_file (file_path)
    except Exception as e:
        row = {
            "filename": file_path.split("/")[-1],
            "filepath": file_path,
            "error": f"{e}",
            "hash": calculate_file_hash(file_path)
        }
        return row

# Define a function that checks whether a certain 
@print_execution_time
def index(
    dir_path = "jari_Hf_measurements"
    #dir_path = "/Volumes/JARI_USB/jari_analog_measurements"
):

    # make short announcement
    print("Indexing current measurements:")

    # Get a list of files to process
    file_paths = []
    for root, dirs, filenames in os.walk(dir_path):
        for file in filenames:
            if file.endswith(".s") and "series" not in file:
                file_paths.append(os.path.join(root, file))

    # Create a progress bar
    progress_bar = tqdm(total=len(file_paths))

    # Create a ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for file_path in file_paths:
            # Submit the process_file function to the executor
            future = executor.submit(process_file_wrapper, file_path)
            # Add the future object to the list
            futures.append(future)

        # Process the completed futures
        for future in concurrent.futures.as_completed(futures):
            # Update the progress bar
            progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    # Concatenate the results into a single DataFrame
    df = pd.DataFrame([future.result() for future in futures if isinstance(future.result(), dict)])

    # postprocess index
    try:
        df = postprocess(df)
    except Exception as e:
        print(e)
    finally:
        # Save the DataFrame to a CSV file
        df.to_csv("index.csv", index=False)

    return df


if __name__ == "__main__":
    print(index())