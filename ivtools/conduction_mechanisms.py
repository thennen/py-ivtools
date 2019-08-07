'''
A collection of functions representing single element electronic conduction mechanisms.

Main reference is:
F.-C. Chiu, “A Review on Conduction Mechanisms in Dielectric Films,” Advances in Materials Science and Engineering, pp. 1–18, 2014.

Each function takes an electric field, temperature, a set of parameters, and returns a current density.

To get I(V) for rectangular prism LxWxt, just do I = mechanism(V/t, ...) * L * W

An attempt is made to provide reasonable default parameters.

Normal numpy broadcasting rules apply, so one can of course pass single parameter values or arrays of values.
'''

from scipy import constants
import numpy as np
from scipy.optimize import fsolve
from functools import reduce, wraps

pi = constants.pi
q = constants.elementary_charge # Because e is 2.71828..
kB = constants.Boltzmann
eps0 = constants.epsilon_0
exp = np.exp
sqrt = np.sqrt


# TODO: Which mechanisms have closed form solutions for E(J, T, ...)? is it useful to include them?
# Maybe a numerical solver can wrap the J(E, T) functions to furnish E(J, T)

# TODO: Give everything default values corresponding to some physical material

# TODO: maybe collect parameters from different materials here so they can be compared, or even passed into conduction mechanism functions ... Maybe can harvest some material parameters from COMSOL..
# i.e.
copper = dict(mu=10,
              Nc=1e7,
              Ef=40,
              Ec=30)
# Now I can call ohmic(E, T, **copper)

# TODO: some first order self-heating model.  What thermal impedances are reasonable?
#       Probably this would look like a self_heating function which takes a mechanism and a thermal impedance

# TODO: linearized plots for the mechanisms that have them

# TODO: Figure out what to do about units.  Want to use electron volts obviously. And many units are commonly given with cm.
# mobility cm^2/V/s, but V/cm is a strange unit for E..
# sometimes choice of units cancel out if you are consistent, but not always?
# can I use some kind of unit manager in python?
# there's a cool package called natu, but says it is pre-release and hasn't been touched in a year

# TODO: Define "physical" ranges for variables.  Could use for brute force or for fit limits

########################################## Bulk limited ##########################################


def ohmic(E, T, mu, Nc, Ec, Ef, **kwargs):
    '''
    E:     Electric Field [Volts/meter]
    T:     Temperature [K]
    mu:    Electron mobility [cm^2/V/s]
    Nc:    Effective density of states in the conduction band [m^-3]
    Ec:    Energy level of conduction band [Volts]
    Ec:    Fermi energy level [eV]
    '''
    # TODO different sets of parameters can be used to specify this ...
    # like resistivity at a certain temperature, or passing n directly, or ...

    # Number of electrons in the conduction band
    n = Nc * exp(-q * (Ec - Ef) / (kB * T))
    # Electrical conductivity
    sigma = n * q * mu
    # Current density
    J = sigma * E
    return J

def poole_frenkel_1d(E, T, mu=4, Nc=5e18, phi_t=0.56, eps_r=20, **kwargs):
    '''
    E:     Electric Field [V/cm]
    T:     Temperature [K]
    mu:    Electron mobility [cm^2/V/s]
    Nc:    Density of states in the conduction band [cm^-3]
    phi_t: Trap energy barrier [eV]
    eps_r: Optical dielectric constant [Unitless]

    Choose any consistent length unit
    '''
    # Energy barrier with electric field lowering
    Eb = q * (phi_t - sqrt(q * E / (pi * eps_r * eps0)))
    kT = kB * T
    J = q * mu * Nc * E * exp(-Eb/kT)
    return J

def poole_frenkel_3d(E, T, sig_p=1.06e4, Ea=0.321, eps_i=9):
    '''

    http://aip.scitation.org/doi/10.1063/1.1655871
    '''
    kT = kB * T
    Ea = Ea * q
    beta = sqrt(q**3 / pi / eps0 / eps_i)
    alpha = beta * sqrt(E) / kT / 2
    J = E * sig_p * exp(-Ea/kT) * (0.25 / alpha**2 * (1 + (alpha - 1) * exp(alpha)) + 0.5)
    return J

#def polaron_hopping(V, n=2e28, a=5e-10, r=8e-10, delta_E=1.23, w=1e17):
def polaron_hopping(E, T, a, b):
    '''
    I am just typing stuff and I don't know if it is right
    An explanation of the parameters
    '''
    kT = kB * T
    return a * E * T**(-3/2) * exp(-b * q/kT)

def mott_gurney():
    '''
    https://en.wikipedia.org/wiki/Space_charge
    '''
    pass

def hopping(E, T, a, n, nu, Ea, ):
    '''
    E:     Electric Field [V/cm]
    T:     Temperature [K]
    a:     Hopping distance [cm]
    n:     Electron concentration in conduction band
    nu:    frequency of thermal vibration of electrons at trap sites [Hz]
    Ea:    Activation energy (Econduction - Etrap) [eV]
    '''
    kT = kB * T
    Eb = (Ea - q * a * E) / kT
    J = q * a * n * nu * exp(-Eb)
    return J


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

######################################## Made up #######################################

def empirical_1(E, T, A=3.38e2, Eb=0.155, c=7.89e-4, **kwargs):
    '''
    E:     Electric Field [V/cm]
    T:     Temperature [K]
    '''
    kT = kB * T
    return E * A * exp(-q*Eb/kT) * exp(c * sqrt(E))


################################### Wrappers #####################################

def macroscopic(mech, A, t, *args, **kwargs):
    # This name might be retarded
    # return a function in terms of V, I instead of j, E
    @wraps(mech)
    def newmech(V, *args, **kwargs):
        return A * mech(V / t, *args, **kwargs)
    return newmech

######################################## Fitting ########################################

def interp_ivt():
    ''' interpolate measured I(V, T) data for uniform I or V values'''

def fit_ivt(i, v, t, thickness, area, mechanism, **kwargs):

    # How to pass the data?
    # i, v, t could be 1d arrays of same length..
    # Usually what is measured I(V) or V(I) for several constant T
    # This is better for plotting as well, because then you can plot lines corresponding to temperatures

    # could pass i and v list of lists, and t just a list

    # At least one of i, v, t needs to be 2d
    # t will probably never be 2d.
    # i and v would both be 2d if the I was not measured at the same V for every array
    # for I(V) loops, V 1d, I 2d, T 1d
    # for V(I) loops, V 2d, I 1d, T 1d
    # then just broadcast the others to the same shape

    # if my normal data structure is passed, i = vstack(df['I']), v = vstack(df['V']), t = df['T']
    pass

######################################## Self-heating #####################################

def T_selfheat(I, V, T0, Rth):
    # This is how temperature depends on J and E, considering self-heating
    # pass I and V and Rth
    # or j and E and rho_th = Rth * volume
    return T0 + Rth * I * V

def self_heating_bulk(J, mechanism, Rth=3.2e5, T0=300, Eguess=1e9, **kwargs):
    '''
    Calculate E(J) if T = T0 + Rth * Power
    passing J values because the result might not be single valued in voltage (s-type NDR)
    How to deal with volume?  maybe pass Rth * volume? do I need to pass A and t??
    Rth has units Kelvin/Watt
    but J*E is Watt/volume, so Rth*J*E is Kelvin/volume
    '''
    # Mechanism should be only missing E and T values
    if kwargs:
        mechanism = reduce(mechanism, **kwargs)
    # Now, given E, we need a consistent solution for I(V, T) and T(I, V)
    E = []
    # For each current density, find the electric field numerically
    for j in J:
        def solvethis(e):
            T = T_selfheat(j, e, T0, Rth)
            jm = mechanism(E=e, T=T)
            return j - jm
        # Volts/meter
        e, *rest = fsolve(solvethis, Eguess)
        E.append(e)
        Eguess = e
    return np.array(E)

######################################## linearized plots #####################################

def schottky_plot():
    pass

def poole_frenkel_plot():
    pass

def arrhenius_plot():
    pass