'''
This is for measuring I(V,T) datasets at the Mark II station using Keithley 2600 and Eurotherm 2408

This is not a standalone script!
You should run it interactively ("%run -i" in ipython) after running the interactive.py script

The script expects:
A Keithley 26xxB connection on variable k
A Eurotherm connection on variable etherm
axes in variables: ax1, ax2, ax3, ax4
'''

# Just checking if the instrument names are defined
assert k is not None
assert etherm is not None

# Plots I want:  I vs V, V/I vs V,
# T vs t should be measuring all the time, every few seconds, with setpoint line
# Some indication of the power, either on its own plot or as a line on IV

temps = np.arange(90, 400, 5):

for T in temps:
    # Set Temperature
    # wait for setpoint while plotting T vs t
    # Measure/save IV
    # Maybe separate loops for low res and high res
    # Plot I vs V, V/I vs V, another point on T vs t. maybe something else

