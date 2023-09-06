"""
It is convenient to have a single function that makes a measurement and returns
the resulting data.

If instrument A delivers the driving signal and instrument B samples the
result, we need to compose the function that:

1. Sets up the scope to capture for the relevant duration (e.g. the waveform duration),
2. Sets up AWG waveform(s) and triggers the AWG (which also triggers the scope capture),
3. Wait until the capture is done and download the data,
4. convert/process the data and return it,

ideally in an instrument-independent way.

e.g. 
picorigoliv = IV(ps.capture, rigol.pulse_arbitrary, ps.get_data, ccircuit_to_iv, **kwargs)

IV(prep_func, trigger_measurement, get_data)?
where these funcs themselves may be compositions?
what is the purpose of IV? is it just IV(a,b,c) = a ∘ b ∘ c ?
probably just to deal with the distribution of arguments and composing a combined signature/docstring
which has nothing to do with IV, actually
but it encodes the general structure of a measurement.
initialize/prepare → start → download results → process results

including what to do with the return values at each step

but then something like interactive_wrapper would want to insert a few additional functions inside that pipeline..
if only in the case of live plotting

Want to impose the least structure on the inner functions as possible

options?
same argument name → shared argument
shared args specified by yet another argument (as interactive_wrapper..)


Question is how to handle the args/kwargs
IV() should expose all the arguments to the inner functions?

combine the ones with the same names?  what about when they have different meanings?
like sample rate for scope and for awg, or channel, which don't have to be the same?
do we just rely on the function definitions to not have the same argument name in that case?

and how to generate a useful docstring/ or at least a function signature?

some of the functions will want to share arguments on purpose (e.g. duration),
and some might accidentally have the same argument names

generated docstrings should make sense somehow


This is actually the same kind of thing I did with interactive_wrapper

which means I will be wrapping a wrapper

It is probably possible to solve this in an even more general way,
where we want to generate composedfunc([list of funcs])

where the funcs don't have to be specially designed for this composition,
but they are called sequentially, and can share arguments / pass data to each other
but we have to define where the data comes from and where it goes..

in python this could be quite clumsy..

Maybe the way you would do it would be to make an IV class with a .run() method...
then just mutate it to do what you want?
you could switch auto saving on/off for example or just insert different functions into the pipeline all willy-nilly
"""

def IV(init_func, start_func, download_func, process_func, **kwargs):
    '''
    Encodes the general structure of a pulse and capture measurement involving separate instruments
    Use to compose single functions that 
    '''
    pass