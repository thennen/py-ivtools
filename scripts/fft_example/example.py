os.chdir('X:/emrl/Pool/Bulletin/Witzleben/py-ivtools/scripts/fft_example')

signal_file = 'Signal_PSPL12500_10ns.txt'
data_signal = pd.read_csv(signal_file, delimiter = '\t')
t_signal = data_signal['t_ttx']
v_signal = data_signal['V_ttx']

measurement_file = 'x01y01_PSPL12500_10ns.txt'
data_meas = pd.read_csv(measurement_file, delimiter = '\t')
t_meas = data_meas['t_ttx']
v_meas = data_meas['V_ttx']

time, v_refl, v_trans = calculate_transmission(file = 'x01y01_kHz.s2p',
t_signal = t_signal,
v_signal = v_signal, 
rf_file = 'x01y01_IPH.s2p',
t_meas = t_meas,
v_meas = v_meas,
do_plots = False,
time_shift = 0,
reflection_offset = 0.27)

v_stimuls = v_signal + v_refl