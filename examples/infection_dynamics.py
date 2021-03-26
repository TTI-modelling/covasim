'''
Demonstrate custom infection dynamics
'''
import numba as nb
import numpy as np
import numpy.random as npr
import covasim as cv
import covasim.infection as cvinf
import covasim.defaults as cvd

# Set dtypes -- note, these cannot be changed after import since Numba functions are precompiled
nbbool  = nb.bool_
nbint   = cvd.nbint
nbfloat = cvd.nbfloat

# define an infectiousness function and compile it in numba for speeeeed
@nb.njit(                        (nbint, nbfloat[:],     nbfloat[:], nbfloat[:])  )
def compute_custom_infectiousness(t,     time_recovered, time_dead,  infectiousness_level): # pragma: no cover
    '''
    Calculate infectiousness for time t.
    Each individual has a different infectiousness level, but infectiousness does not
    depend on time

    Args:
        t: (int) timestep
        time_start: (float[]) individuals' infectious date
        time_recovered: (float[]) individuals' recovered date
        time_dead: (float[]) individuals' death date
        infectiousness_level: (float[]) individuals infectiousness parameters, one per individual

    Returns:
        infectiousness (float): infectiousness
    '''

    # Get the end date from recover or death
    n = len(time_dead)
    time_stop = np.ones(n, dtype=cvd.default_float)*time_recovered # This is needed to make a copy
    inds = ~np.isnan(time_dead)
    time_stop[inds] = time_dead[inds]
    # time_stop is effectively min(time_recovered, time_dead), which can be in the future

    # who is still infectious? i.e. hasn't recovered
    infectious_individuals_index = t < time_stop 

    # create an array of length n containing only zeros
    infectiousness = np.zeros(n, dtype=cvd.default_float)

    # for the infectious individuals, set their infectiousness equal to their infectiousness level
    # which is a parameter they are given at time 0
    infectiousness[infectious_individuals_index] = infectiousness_level[infectious_individuals_index]
    infectiousness[~infectious_individuals_index] = 0.0

    return infectiousness

class uniform_infectiousness_with_heterogeneity(cvinf.InfectionDynamics):
    '''
    An infectiousness profile where infectiousness does not depend on your infectious age,
    but does vary between individuals
    '''

    def initialize(self, sim: cv.Sim):
        
        # We give each individual an infectiousness_level
        sim.people.infectiousness_level = npr.uniform(low = 0.1, high = 2.0, size = sim['pop_size'])
        sim.people.infectiousness_level = sim.people.infectiousness_level.astype(cvd.default_float)

    def compute_infectiousness(self, sim):
        
        date_recovered       = sim.people.date_recovered
        date_dead            = sim.people.date_dead
        infectiousness_level = sim.people.infectiousness_level 
        t                    = sim.t

        return compute_custom_infectiousness(t, date_recovered, date_dead, infectiousness_level)

if __name__ == '__main__':

    # Define and run the baseline simulation
    pars = dict(
        pop_size = 50e3,
        pop_infected = 100,
        n_days = 90,
        verbose = 0,
    )
    orig_sim = cv.Sim(pars, label='Default')

    # create an instance of the infection dynamics
    custom_infection_model = uniform_infectiousness_with_heterogeneity()
    sim = cv.Sim(pars, infection_dynamics=custom_infection_model, label='Custom infection dynamics')

    # Run and plot
    msim = cv.MultiSim([orig_sim, sim])
    msim.run()
    msim.plot()
