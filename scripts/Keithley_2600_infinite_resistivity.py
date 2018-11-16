# Run interactive.py before this

# This is for measuring 4 pt resistivity over the weekend to test for drift
# Will write one file per IV loop until you press ctrl-C
# Plots resistance vs cycle, and the last 20 IV loops

iplots.plotters = keithley_plotters
# So program doesn't slow down over time, and we don't want to see so many curves anyway
iplots.set_maxlines(20)

iplots.createfig(4)
iplots.add_plotter(R_vs_cycle_plotter, 4)

# Resistivity modules are 22, 23
#meta.load_lassen(module=22)
meta.meta = dict(sample_name='lol',
                 module='022')

irange = 1e-6
ilim = 1e-5
npts = 200
nplc = 1
wfm = tri(.2, -.2, npts)

show()

while True:
    # will save and plot automatically
    d = kiv_4pt(wfm, irange, ilim, nplc)
