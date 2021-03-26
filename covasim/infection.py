'''
Specify a class that allows users to apply custom infection dynamics
'''
from . import defaults as cvd
from . import utils as cvu


class InfectionDynamics():

    '''
    Base class for infection dynamics.
    '''

    def __init__(self, label=None) -> None:
        
        self.label = label # e.g. "Martyn's infection dynamics model"
        return
    
    def initialize(self, sim):
        '''
        Initialize attributes that will be used when calculating infectiousness

        For example, some individuals may be super-sneezers, and have a higher level of relative transmissibility.
        This can be decided at the beginning of the simulation
        '''
        self.initialized = True
        return

    def compute_relative_transmissibility(self, sim):
        '''
        Takes the simulation object as an input and compute the relative transmissibility for infectious nodes.

        Usually gets called every time step

        Args:
            sim: The Sim instance

        Returns:
            None
        '''
        raise NotImplementedError

class HighLowInfectiousness(InfectionDynamics):

    def compute_relative_transmissibility(self, sim):
        '''
        Computes relative transmissibility using a default implementation of infectiousness
        '''
        
        # unpack relevent parameters from this step of the simulation
        frac_time    = cvd.default_float(sim['viral_dist']['frac_time'])
        load_ratio   = cvd.default_float(sim['viral_dist']['load_ratio'])
        high_cap     = cvd.default_float(sim['viral_dist']['high_cap'])
        date_inf     = sim.people.date_infectious
        date_rec     = sim.people.date_recovered
        date_dead    = sim.people.date_dead
        t            = sim.t

        # use the implementation of the transmission dynamics implemented in the base model in Numba
        return cvu.high_low_transmissibility(t, date_inf, date_rec, date_dead, frac_time, load_ratio, high_cap)
