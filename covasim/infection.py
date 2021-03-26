'''
Specify a class that allows users to apply custom infection dynamics
'''
class InfectionDynamics():

    '''
    Base class for infection dynamics.
    '''

    def __init__(self, label=None) -> None:
        
        self.label = label # e.g. "On-off viral load"
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

        Args:
            sim: The Sim instance

        Returns:
            None
        '''
        raise NotImplementedError
