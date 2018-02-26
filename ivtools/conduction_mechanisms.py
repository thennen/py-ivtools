'''
A collection of functions representing single element electronic conduction mechanisms.

Each function takes an applied voltage, a set of parameters, and returns a current.

An attempt is made to provide reasonable default parameters.

Normal numpy broadcasting rules apply, so one can of course pass single parameter values or arrays of values.

All units should be in SI?
'''

########################################## Bulk limited ##########################################

def resistor(V, T, R_0=1e6, T_0=300, alpha=0):
    '''
    V:     Applied voltage [Volts]
    T:     Resistor temperature [Kelvin]
    R_0:   Resistance at given temperature [Ohm]
    T_0:   The temperature at which the resistance is R_0 [Kelvin]
    alpha: The temperature coefficient of resistance [1/Kelvin]
    '''
    return V / (R + alpha * (T - T0))

def poole_frenkel_1d(V, d=6e-9, C=6e-15, PhiB=0.1, eps_i=26, d_z=1e-10, mu_n=1800):
    '''
    An explanation of the parameters
    '''
    pass

def poole_frenkel_3d():
    '''
    http://aip.scitation.org/doi/10.1063/1.1655871
    '''
    pass

def polaron_hopping(V, n=2e28, a=5e-10, r=8e-10, delta_E=1.23, w=1e17):
    '''
    An explanation of the parameters
    '''
    pass

def mott_gurney():
    '''
    https://en.wikipedia.org/wiki/Space_charge
    '''
    pass

def hopping():
    pass

######################################## Interface limited ########################################

def fowler_nordheim():
    '''
    http://www.ece.mcmaster.ca/~chihhung/Publication/MR_46_Gate_Tunneling.pdf
    '''
    pass

def direct_tunneling():
    '''
    http://www.iue.tuwien.ac.at/phd/gehring/node60.html
    '''
    pass

def thermionic_emission():
    '''
    '''
    pass

def field_emission():
    '''
    '''
    pass

def thermionic_field_emission():
    '''
    '''
    pass
