'''
Defines the Person class and functions associated with making people.
'''

#%% Imports
from typing import Type
import numpy as np
import sciris as sc
from collections import defaultdict
from . import version as cvv
from . import utils as cvu
from . import defaults as cvd
from . import base as cvb
from . import plotting as cvplt
from . import immunity as cvi
from .test_sensitivity_profile import TestSensitivityProfile, BinomialTestSensitivityProfile


__all__ = ['People']

class People(cvb.BasePeople):
    '''
    A class to perform all the operations on the people. This class is usually
    not invoked directly, but instead is created automatically by the sim. The
    only required input argument is the population size, but typically the full
    parameters dictionary will get passed instead since it will be needed before
    the People object is initialized.

    Note that this class handles the mechanics of updating the actual people, while
    BasePeople takes care of housekeeping (saving, loading, exporting, etc.). Please
    see the BasePeople class for additional methods.

    Args:
        pars (dict): the sim parameters, e.g. sim.pars -- alternatively, if a number, interpreted as pop_size
        strict (bool): whether or not to only create keys that are already in self.meta.person; otherwise, let any key be set
        kwargs (dict): the actual data, e.g. from a popdict, being specified

    ::Examples::

        ppl1 = cv.People(2000)

        sim = cv.Sim()
        ppl2 = cv.People(sim.pars)
    '''

    def __init__(self, pars, strict=True, **kwargs):


        #object = GetTestSensitivityCurve ('./data/pcr_pars.csv', './data/lfa_pars.csv')
        #self.lfa = object.lfa_prob_positive
        #self.pcr=object.pcr_prob_positive


        # Handle pars and population size
        self.set_pars(pars)
        self.version = cvv.__version__ # Store version info

        # try to get available test sensitivity profiles, set to None otherwise
        self.test_sensitivity_profiles = pars.get('test_sensitivity_profiles')

        # Other initialization
        self.t = 0 # Keep current simulation time
        self._lock = False # Prevent further modification of keys
        self.meta = cvd.PeopleMeta() # Store list of keys and dtypes
        self.contacts = None
        self.init_contacts() # Initialize the contacts
        self.infection_log = [] # Record of infections - keys for ['source','target','date','layer']

        # Set person properties -- all floats except for UID
        for key in self.meta.person:
            if key == 'uid':
                self[key] = np.arange(self.pars['pop_size'], dtype=cvd.default_int)
            else:
                self[key] = np.full(self.pars['pop_size'], np.nan, dtype=cvd.default_float)

        # Set health states -- only susceptible is true by default -- booleans except exposed by strain which should return the strain that ind is exposed to
        for key in self.meta.states:
            val = (key in ['susceptible', 'naive']) # Default value is True for susceptible and naive, false otherwise
            self[key] = np.full(self.pars['pop_size'], val, dtype=bool)

        # Set strain states, which store info about which strain a person is exposed to
        for key in self.meta.strain_states:
            self[key] = np.full(self.pars['pop_size'], np.nan, dtype=cvd.default_float)
        for key in self.meta.by_strain_states:
            self[key] = np.full((self.pars['n_strains'], self.pars['pop_size']), False, dtype=bool)

        # Set immunity and antibody states
        for key in self.meta.imm_states:  # Everyone starts out with no immunity
            self[key] = np.zeros((self.pars['n_strains'], self.pars['pop_size']), dtype=cvd.default_float)
        for key in self.meta.nab_states:  # Everyone starts out with no antibodies
            self[key] = np.full(self.pars['pop_size'], np.nan, dtype=cvd.default_float)
        for key in self.meta.vacc_states:
            self[key] = np.zeros(self.pars['pop_size'], dtype=cvd.default_int)

        # Set dates and durations -- both floats
        for key in self.meta.dates + self.meta.durs:
            self[key] = np.full(self.pars['pop_size'], np.nan, dtype=cvd.default_float)

        # Store the dtypes used in a flat dict
        self._dtypes = {key:self[key].dtype for key in self.keys()} # Assign all to float by default
        if strict:
            self.lock() # If strict is true, stop further keys from being set (does not affect attributes)

        # Store flows to be computed during simulation
        self.init_flows()

        # Although we have called init(), we still need to call initialize()
        self.initialized = False

        # Handle contacts, if supplied (note: they usually are)
        if 'contacts' in kwargs:
            self.add_contacts(kwargs.pop('contacts'))

        # Handle all other values, e.g. age
        for key,value in kwargs.items():
            if strict:
                self.set(key, value)
            else:
                self[key] = value

        self._pending_quarantine = defaultdict(list)  # Internal cache to record people that need to be quarantined on each timestep {t:(person, quarantine_end_day)}

        self._pending_isolation = defaultdict(list)

        #self.new_wrong_quarantines = np.full(181, 0, dtype=cvd.default_int)
        self.tests_remaining = np.full(self.pars['pop_size'], np.nan)
        self.date_ili_recover = np.full(self.pars['pop_size'], np.nan)
        return


    def init_flows(self):
        ''' Initialize flows to be zero '''
        self.flows = {key:0 for key in cvd.new_result_flows}
        self.flows_strain = {}
        for key in cvd.new_result_flows_by_strain:
            self.flows_strain[key] = np.zeros(self.pars['n_strains'], dtype=cvd.default_float)
        return


    def initialize(self):
        ''' Perform initializations '''
        self.set_prognoses()
        self.validate()
        self.initialized = True

        # initialize the test sensitivity profiles
        for profile in self.test_sensitivity_profiles.values():

            if not issubclass(type(profile), TestSensitivityProfile):
                print(profile)
                print(TestSensitivityProfile)
                raise TypeError("Profile is not a test sensitivity profile")

            # initialise the profile
            profile.initialize(self)

        return


    def set_prognoses(self):
        '''
        Set the prognoses for each person based on age during initialization. Need
        to reset the seed because viral loads are drawn stochastically.
        '''

        pars = self.pars # Shorten
        if 'prognoses' not in pars or 'rand_seed' not in pars:
            errormsg = 'This people object does not have the required parameters ("prognoses" and "rand_seed"). Create a sim (or parameters), then do e.g. people.set_pars(sim.pars).'
            raise sc.KeyNotFoundError(errormsg)

        def find_cutoff(age_cutoffs, age):
            '''
            Find which age bin each person belongs to -- e.g. with standard
            age bins 0, 10, 20, etc., ages [5, 12, 4, 58] would be mapped to
            indices [0, 1, 0, 5]. Age bins are not guaranteed to be uniform
            width, which is why this can't be done as an array operation.
            '''
            return np.nonzero(age_cutoffs <= age)[0][-1]  # Index of the age bin to use

        cvu.set_seed(pars['rand_seed'])

        progs = pars['prognoses'] # Shorten the name
        inds = np.fromiter((find_cutoff(progs['age_cutoffs'], this_age) for this_age in self.age), dtype=cvd.default_int, count=len(self)) # Convert ages to indices
        self.symp_prob[:]   = progs['symp_probs'][inds] # Probability of developing symptoms
        self.severe_prob[:] = progs['severe_probs'][inds]*progs['comorbidities'][inds] # Severe disease probability is modified by comorbidities
        self.crit_prob[:]   = progs['crit_probs'][inds] # Probability of developing critical disease
        self.death_prob[:]  = progs['death_probs'][inds] # Probability of death
        self.rel_sus[:]     = progs['sus_ORs'][inds]  # Default susceptibilities
        self.rel_trans[:]   = progs['trans_ORs'][inds] * cvu.sample(**self.pars['beta_dist'], size=len(inds))  # Default transmissibilities, with viral load drawn from a distribution

        return


    def update_states_pre(self, t):
        ''' Perform all state updates at the current timestep '''

        # Initialize
        self.t = t
        self.is_exp = self.true('exposed') # For storing the interim values since used in every subsequent calculation

        # Perform updates
        self.init_flows()
        self.flows['new_infectious']  += self.check_infectious() # For people who are exposed and not infectious, check if they begin being infectious
        self.flows['new_symptomatic'] += self.check_symptomatic()
        self.flows['new_severe']      += self.check_severe()
        self.flows['new_critical']    += self.check_critical()
        self.flows['new_deaths']      += self.check_death()
        self.flows['new_recoveries']  += self.check_recovery() # TODO: check logic here

        return


    def update_states_post(self):
        ''' Perform post-timestep updates '''
        self.flows['new_diagnoses']   += self.check_diag()
        self.flows['new_quarantined'] += self.check_quar()
        del self.is_exp  # Tidy up

        return


    def update_contacts(self):
        ''' Refresh dynamic contacts, e.g. community '''
        # Figure out if anything needs to be done -- e.g. {'h':False, 'c':True}
        for lkey, is_dynam in self.pars['dynam_layer'].items():
            if is_dynam:
                if self.pars["pop_type"] == "matrix":
                    self.contacts[lkey].update_agemixing(self)
                else:
                    self.contacts[lkey].update(self)

        return self.contacts


    #%% Methods for updating state

    def check_inds(self, current, date, filter_inds=None):
        ''' Return indices for which the current state is false and which meet the date criterion '''
        if filter_inds is None:
            not_current = cvu.false(current)
        else:
            not_current = cvu.ifalsei(current, filter_inds)
        has_date = cvu.idefinedi(date, not_current)
        inds     = cvu.itrue(self.t >= date[has_date], has_date)
        return inds


    def check_infectious(self):
        ''' Check if they become infectious '''
        inds = self.check_inds(self.infectious, self.date_infectious, filter_inds=self.is_exp)
        self.infectious[inds] = True
        self.infectious_strain[inds] = self.exposed_strain[inds]
        for strain in range(self.pars['n_strains']):
            this_strain_inds = cvu.itrue(self.infectious_strain[inds] == strain, inds)
            n_this_strain_inds = len(this_strain_inds)
            self.flows_strain['new_infectious_by_strain'][strain] += n_this_strain_inds
            self.infectious_by_strain[strain, this_strain_inds] = True
        return len(inds)


    def check_symptomatic(self):
        ''' Check for new progressions to symptomatic '''
        inds = self.check_inds(self.symptomatic, self.date_symptomatic, filter_inds=self.is_exp)
        self.symptomatic[inds] = True
        return len(inds)


    def check_severe(self):
        ''' Check for new progressions to severe '''
        inds = self.check_inds(self.severe, self.date_severe, filter_inds=self.is_exp)
        self.severe[inds] = True
        return len(inds)


    def check_critical(self):
        ''' Check for new progressions to critical '''
        inds = self.check_inds(self.critical, self.date_critical, filter_inds=self.is_exp)
        self.critical[inds] = True
        return len(inds)


    def check_recovery(self, inds=None, filter_inds='is_exp'):
        '''
        Check for recovery.

        More complex than other functions to allow for recovery to be manually imposed
        for a specified set of indices.
        '''

        # Handle more flexible options for setting indices
        if filter_inds == 'is_exp':
            filter_inds = self.is_exp
        if inds is None:
            inds = self.check_inds(self.recovered, self.date_recovered, filter_inds=filter_inds)

        # Now reset all disease states
        self.exposed[inds]          = False
        self.infectious[inds]       = False
        self.symptomatic[inds]      = False
        self.severe[inds]           = False
        self.critical[inds]         = False
        self.recovered[inds]        = True
        self.recovered_strain[inds] = self.exposed_strain[inds]
        self.infectious_strain[inds] = np.nan
        self.exposed_strain[inds]    = np.nan
        self.exposed_by_strain[:, inds] = False
        self.infectious_by_strain[:, inds] = False


        # Handle immunity aspects
        if self.pars['use_waning']:

            # Before letting them recover, store information about the strain they had, store symptoms and pre-compute nabs array
            mild_inds   = self.check_inds(self.susceptible, self.date_symptomatic, filter_inds=inds)
            severe_inds = self.check_inds(self.susceptible, self.date_severe,      filter_inds=inds)

            # Reset additional states
            self.susceptible[inds] = True
            self.diagnosed[inds]   = False # Reset their diagnosis state because they might be reinfected
            self.prior_symptoms[inds]        = self.pars['rel_imm_symp']['asymp']
            self.prior_symptoms[mild_inds]   = self.pars['rel_imm_symp']['mild']
            self.prior_symptoms[severe_inds] = self.pars['rel_imm_symp']['severe']
            if len(inds):
                cvi.init_nab(self, inds, prior_inf=True)

        return len(inds)


    def check_death(self):
        ''' Check whether or not this person died on this timestep  '''
        inds = self.check_inds(self.dead, self.date_dead, filter_inds=self.is_exp)
        self.susceptible[inds]   = False
        self.exposed[inds]       = False
        self.infectious[inds]    = False
        self.symptomatic[inds]   = False
        self.severe[inds]        = False
        self.critical[inds]      = False
        self.known_contact[inds] = False
        self.quarantined[inds]   = False
        self.recovered[inds]     = False
        self.dead[inds]          = True
        self.infectious_strain[inds] = np.nan
        self.exposed_strain[inds]    = np.nan
        self.recovered_strain[inds]  = np.nan
        return len(inds)


    def check_diagnosed(self):
        '''
        Check for new diagnoses. Since most data are reported with diagnoses on
        the date of the test, this function reports counts not for the number of
        people who received a positive test result on a day, but rather, the number
        of people who were tested on that day who are schedule to be diagnosed in
        the future.
        '''

        # Handle people who tested today who will be diagnosed in future
        test_pos_inds = self.check_inds(self.diagnosed, self.date_pos_test, filter_inds=None) # Find people who will be diagnosed in future
        self.date_pos_test[test_pos_inds] = np.nan # Clear date of having will-be-positive test

        # Handle people who were actually diagnosed today
        diag_inds  = self.check_inds(self.diagnosed, self.date_diagnosed, filter_inds=None) # Find who was actually diagnosed on this timestep
        self.diagnosed[diag_inds]   = True # Set these people to be diagnosed
        quarantined = cvu.itruei(self.quarantined, diag_inds)
        self.date_end_quarantine[quarantined] = self.t # Set end quarantine date to match when the person left quarantine (and entered isolation)
        self.quarantined[diag_inds] = False # If you are diagnosed, you are isolated, not in quarantine
        self.ttr[diag_inds] = False

        return len(test_pos_inds)


    def check_diag(self):
        ''' Update diagnosis state, same way as quarantine '''

        n_diagnosed = 0
        for ind, end_day in self._pending_isolation[self.t]:
            if self.diagnosed[ind]:
                self.date_end_diagnosis[ind] = max(self.date_end_diagnosis[ind], end_day)
            elif not self.dead[ind]:
                self.diagnosed[ind] = True
                self.date_end_diagnosis[ind] = end_day
                self.quarantined[ind] = False
                self.date_end_quarantine[ind] = self.t
                n_diagnosed += 1


        end_inds = self.check_inds(~self.diagnosed, self.date_end_diagnosis, filter_inds=None)
        self.diagnosed[end_inds] = False
        return n_diagnosed

    def check_quar(self):
        ''' Update quarantine state '''

        n_quarantined = 0 # Number of people entering quarantine
        for ind,end_day in self._pending_quarantine[self.t]:
            if self.quarantined[ind]:
                self.date_end_quarantine[ind] = max(self.date_end_quarantine[ind], end_day) # Extend quarantine if required
                self.date_quarantined[ind] = self.t
            elif not (self.dead[ind] or self.diagnosed[ind]): # Unclear whether recovered should be included here
                self.quarantined[ind] = True
                self.date_quarantined[ind] = self.t
                self.date_end_quarantine[ind] = end_day
                n_quarantined += 1

        # If someone on quarantine has reached the end of their quarantine, release them
        end_inds = self.check_inds(~self.quarantined, self.date_end_quarantine, filter_inds=None) # Note the double-negative here (~)
        self.quarantined[end_inds] = False # Release from quarantine

        return n_quarantined


    #%% Methods to make events occur (infection and diagnosis)

    def make_naive(self, inds):
        '''
        Make a set of people naive. This is used during dynamic resampling.
        '''
        for key in self.meta.states:
            if key in ['susceptible', 'naive']:
                self[key][inds] = True
            else:
                self[key][inds] = False

        # Reset strain states
        for key in self.meta.strain_states:
            self[key][inds] = np.nan
        for key in self.meta.by_strain_states:
            self[key][:, inds] = False

        # Reset immunity and antibody states
        for key in self.meta.imm_states:
            self[key][:, inds] = 0
        for key in self.meta.nab_states:
            self[key][inds] = np.nan
        for key in self.meta.vacc_states:
            self[key][inds] = 0

        # Reset dates
        for key in self.meta.dates + self.meta.durs:
            self[key][inds] = np.nan

        return


    def make_nonnaive(self, inds, set_recovered=False, date_recovered=0):
        '''
        Make a set of people non-naive.

        This can be done either by setting only susceptible and naive states,
        or else by setting them as if they have been infected and recovered.
        '''
        self.make_naive(inds) # First make them naive and reset all other states

        # Make them non-naive
        for key in ['susceptible', 'naive']:
            self[key][inds] = False

        if set_recovered:
            self.date_recovered[inds] = date_recovered # Reset date recovered
            self.check_recovered(inds=inds, filter_inds=None) # Set recovered

        return



    def infect(self, inds, hosp_max=None, icu_max=None, source=None, layer=None, strain=0):
        '''
        Infect people and determine their eventual outcomes.

            * Every infected person can infect other people, regardless of whether they develop symptoms
            * Infected people that develop symptoms are disaggregated into mild vs. severe (=requires hospitalization) vs. critical (=requires ICU)
            * Every asymptomatic, mildly symptomatic, and severely symptomatic person recovers
            * Critical cases either recover or die

        Method also deduplicates input arrays in case one agent is infected many times
        and stores who infected whom in infection_log list.

        Args:
            inds     (array): array of people to infect
            hosp_max (bool):  whether or not there is an acute bed available for this person
            icu_max  (bool):  whether or not there is an ICU bed available for this person
            source   (array): source indices of the people who transmitted this infection (None if an importation or seed infection)
            layer    (str):   contact layer this infection was transmitted on
            strain   (int):   the strain people are being infected by

        Returns:
            count (int): number of people infected
        '''

        # Remove duplicates
        inds, unique = np.unique(inds, return_index=True)
        if source is not None:
            source = source[unique]

        # Keep only susceptibles
        keep = self.susceptible[inds] # Unique indices in inds and source that are also susceptible
        inds = inds[keep]
        if source is not None:
            source = source[keep]

        if self.pars['use_waning']:
            cvi.check_immunity(self, strain, sus=False, inds=inds)

        # Deal with strain parameters
        strain_keys = ['rel_symp_prob', 'rel_severe_prob', 'rel_crit_prob', 'rel_death_prob']
        infect_pars = {k:self.pars[k] for k in strain_keys}
        if strain:
            strain_label = self.pars['strain_map'][strain]
            for k in strain_keys:
                infect_pars[k] *= self.pars['strain_pars'][strain_label][k]

        n_infections = len(inds)
        durpars      = self.pars['dur']

        # Update states, strain info, and flows
        self.susceptible[inds]    = False
        self.naive[inds]          = False
        self.recovered[inds]      = False
        self.diagnosed[inds]      = False
        self.exposed[inds]        = True
        self.exposed_strain[inds] = strain
        self.exposed_by_strain[strain, inds] = True
        self.flows['new_infections']   += len(inds)
        self.flows['new_reinfections'] += len(cvu.defined(self.date_recovered[inds])) # Record reinfections
        self.flows_strain['new_infections_by_strain'][strain] += len(inds)
        # print('HI DEBUG', self.t, len(inds), len(cvu.defined(self.date_recovered[inds])))
        # self.date_recovered[inds] = np.nan # Reset date they recovered - we only store the last recovery # TODO CK

        # Record transmissions
        for i, target in enumerate(inds):
            self.infection_log.append(dict(source=source[i] if source is not None else None, target=target, date=self.t, layer=layer))

        # Calculate how long before this person can infect other people
        #self.dur_exp2inf[inds] = cvu.sample(**durpars['exp2inf'], size=n_infections)
        self.date_exposed[inds]   = self.t
        self.date_infectious[inds] = self.t#self.dur_exp2inf[inds] + self.t

        # Reset all other dates
        for key in ['date_symptomatic', 'date_severe', 'date_critical', 'date_diagnosed', 'date_recovered']:
            self[key][inds] = np.nan

        # Use prognosis probabilities to determine what happens to them
        symp_probs = infect_pars['rel_symp_prob']*self.symp_prob[inds]*(1-self.symp_imm[strain, inds]) # Calculate their actual probability of being symptomatic
        is_symp = cvu.binomial_arr(symp_probs) # Determine if they develop symptoms
        symp_inds = inds[is_symp]
        asymp_inds = inds[~is_symp] # Asymptomatic

        # CASE 1: Asymptomatic: may infect others, but have no symptoms and do not die
        #dur_asym2rec = cvu.sample(**durpars['asym2rec'], size=len(asymp_inds))
        self.date_recovered[asymp_inds] = self.date_infectious[asymp_inds] + 14#dur_asym2rec  # Date they recover
        self.dur_disease[asymp_inds] = 14 #self.dur_exp2inf[asymp_inds] + dur_asym2rec  # Store how long this person had COVID-19

        # CASE 2: Symptomatic: can either be mild, severe, or critical
        n_symp_inds = len(symp_inds)
        self.dur_inf2sym[symp_inds] = cvu.sample(**durpars['inf2sym'], size=n_symp_inds) # Store how long this person took to develop symptoms
        self.date_symptomatic[symp_inds] = self.date_infectious[symp_inds] + self.dur_inf2sym[symp_inds] # Date they become symptomatic
        sev_probs = infect_pars['rel_severe_prob'] * self.severe_prob[symp_inds]*(1-self.sev_imm[strain, symp_inds]) # Probability of these people being severe
        # print(self.sev_imm[strain, inds])
        is_sev = cvu.binomial_arr(sev_probs) # See if they're a severe or mild case
        sev_inds = symp_inds[is_sev]
        mild_inds = symp_inds[~is_sev] # Not severe

        # CASE 2.1: Mild symptoms, no hospitalization required and no probability of death
        #dur_mild2rec = cvu.sample(**durpars['mild2rec'], size=len(mild_inds))
        self.date_recovered[mild_inds] = self.date_symptomatic[mild_inds] + 14#dur_mild2rec  # Date they recover
        self.dur_disease[mild_inds] = self.dur_inf2sym[mild_inds] + 14 #self.dur_exp2inf[mild_inds] + self.dur_inf2sym[mild_inds] + dur_mild2rec  # Store how long this person had COVID-19

        # CASE 2.2: Severe cases: hospitalization required, may become critical
        self.dur_sym2sev[sev_inds] = cvu.sample(**durpars['sym2sev'], size=len(sev_inds)) # Store how long this person took to develop severe symptoms
        self.date_severe[sev_inds] = self.date_symptomatic[sev_inds] + self.dur_sym2sev[sev_inds]  # Date symptoms become severe
        crit_probs = infect_pars['rel_crit_prob'] * self.crit_prob[sev_inds] * (self.pars['no_hosp_factor'] if hosp_max else 1.) # Probability of these people becoming critical - higher if no beds available
        is_crit = cvu.binomial_arr(crit_probs)  # See if they're a critical case
        crit_inds = sev_inds[is_crit]
        non_crit_inds = sev_inds[~is_crit]

        # CASE 2.2.1 Not critical - they will recover
        dur_sev2rec = cvu.sample(**durpars['sev2rec'], size=len(non_crit_inds))
        self.date_recovered[non_crit_inds] = self.date_severe[non_crit_inds] + dur_sev2rec  # Date they recover
        self.dur_disease[non_crit_inds] = self.dur_inf2sym[non_crit_inds] + self.dur_sym2sev[non_crit_inds] + dur_sev2rec #self.dur_exp2inf[non_crit_inds] + self.dur_inf2sym[non_crit_inds] + self.dur_sym2sev[non_crit_inds] + dur_sev2rec  # Store how long this person had COVID-19

        # CASE 2.2.2: Critical cases: ICU required, may die
        self.dur_sev2crit[crit_inds] = cvu.sample(**durpars['sev2crit'], size=len(crit_inds))
        self.date_critical[crit_inds] = self.date_severe[crit_inds] + self.dur_sev2crit[crit_inds]  # Date they become critical
        death_probs = infect_pars['rel_death_prob'] * self.death_prob[crit_inds] * (self.pars['no_icu_factor'] if icu_max else 1.)# Probability they'll die
        is_dead = cvu.binomial_arr(death_probs)  # Death outcome
        dead_inds = crit_inds[is_dead]
        alive_inds = crit_inds[~is_dead]

        # CASE 2.2.2.1: Did not die
        dur_crit2rec = cvu.sample(**durpars['crit2rec'], size=len(alive_inds))
        self.date_recovered[alive_inds] = self.date_critical[alive_inds] + dur_crit2rec # Date they recover
        self.dur_disease[alive_inds] = self.dur_inf2sym[alive_inds] + self.dur_sym2sev[alive_inds] + self.dur_sev2crit[alive_inds] + dur_crit2rec #self.dur_exp2inf[alive_inds] + self.dur_inf2sym[alive_inds] + self.dur_sym2sev[alive_inds] + self.dur_sev2crit[alive_inds] + dur_crit2rec  # Store how long this person had COVID-19

        # CASE 2.2.2.2: Did die
        dur_crit2die = cvu.sample(**durpars['crit2die'], size=len(dead_inds))
        self.date_dead[dead_inds] = self.date_critical[dead_inds] + dur_crit2die # Date of death
        self.dur_disease[dead_inds] = self.dur_inf2sym[dead_inds] + self.dur_sym2sev[dead_inds] + self.dur_sev2crit[dead_inds] + dur_crit2die #self.dur_exp2inf[dead_inds] + self.dur_inf2sym[dead_inds] + self.dur_sym2sev[dead_inds] + self.dur_sev2crit[dead_inds] + dur_crit2die   # Store how long this person had COVID-19
        self.date_recovered[dead_inds] = np.nan # If they did die, remove them from recovered

        return n_infections # For incrementing counters


    def test_custom_profile(self, inds, test_sensitivity_profile_key: str):
        """
        Works similarly to the standard testing function, with the exception that it
        calls a test_sensitivity_profile. This allows different types of tests to be used
        in the same simulation.

        Test sensitivity profiles need to be passed as part of pars, in a dictionary format.

        To use a specific profile, set the test_sensitivity_profile_key to the value in the
        dictionary.

        Args:
            test_sensitivity_profile (str): the key of the desired test sensitivity profile
        """

        profile = self.test_sensitivity_profiles[test_sensitivity_profile_key]

        profile.test(self, inds)

        return


    def test(self, inds, test_sensitivity=1.0, loss_prob=0.0, test_delay=0):
        '''
        Method to test people. Typically not to be called by the user directly;
        see the test_num() and test_prob() interventions.

        Args:
            inds: indices of who to test
            test_sensitivity (float): probability of a true positive
            loss_prob (float): probability of loss to follow-up
            test_delay (int): number of days before test results are ready
        '''

        inds = np.unique(inds)
        self.tested[inds] = True
        self.date_tested[inds] = self.t # Only keep the last time they tested

        is_infectious = cvu.itruei(self.infectious, inds)
        pos_test      = cvu.n_binomial(test_sensitivity, len(is_infectious))
        is_inf_pos    = is_infectious[pos_test]

        not_diagnosed = is_inf_pos[np.isnan(self.date_diagnosed[is_inf_pos])]
        not_lost      = cvu.n_binomial(1.0-loss_prob, len(not_diagnosed))
        final_inds    = not_diagnosed[not_lost]

        # Store the date the person will be diagnosed, as well as the date they took the test which will come back positive
        self.date_diagnosed[final_inds] = self.t + test_delay
        self.date_pos_test[final_inds] = self.t

        return


    def test_specificity(self, inds, test_sensitivity=1.0, loss_prob=0.0, test_delay=0, test_specif = 1.0):
        '''
        Not in use anymore.
        Method to test people with false positives. Typically not to be called by the user directly;
        see the test_num() and test_prob() interventions.

        Args:
            inds: indices of who to test
            test_sensitivity (float): probability of a true positive
            loss_prob (float): probability of loss to follow-up
            test_delay (int): number of days before test results are ready
        '''

        inds = np.unique(inds)
        self.tested[inds] = True
        self.date_tested[inds] = self.t # Only keep the last time they tested

        #bad_swab = cvu.n_binomial(swab_error_rate, len(inds))
        #bad_swab_negatives = inds[bad_swab]

        is_infectious = cvu.itruei(self.infectious, inds)
        pos_test      = cvu.n_binomial(test_sensitivity, len(is_infectious))
        is_inf_pos    = is_infectious[pos_test]

        is_well = cvu.ifalsei(self.infectious, inds)
        pos_test = cvu.n_binomial(1-test_specif, len(is_well))
        is_well_pos = is_well[pos_test]

        all_positives = np.unique(np.concatenate([is_inf_pos,is_well_pos]))
        not_lost      = cvu.n_binomial(1.0-loss_prob, len(all_positives))
        final_inds    = all_positives[not_lost]
        #final_inds = final_inds[~np.in1d(final_inds,bad_swab_negatives)]

        # Store the date the person will be diagnosed, as well as the date they took the test which will come back positive
        self.date_diagnosed[final_inds] = self.t + test_delay
        self.date_pos_test[final_inds] = self.t

        self.schedule_isolation(final_inds, start_date = self.t + test_delay)
        return


    def schedule_isolation(self, inds, start_date = None, period = None):
        if not isinstance(start_date, np.ndarray):
            start_date = self.t if start_date is None else int(start_date)
            start_date = np.array([start_date] * len(inds))
        period = self.pars['iso_period'] if period is None else int(period)
        for i,ind in enumerate(inds):
            self._pending_isolation[start_date[i]].append((ind, start_date[i] + period))
        return

    def schedule_quarantine(self, inds, start_date=None, period=None):
        '''
        Schedule a quarantine. Typically not called by the user directly except
        via a custom intervention; see the contact_tracing() intervention instead.

        This function will create a request to quarantine a person on the start_date for
        a period of time. Whether they are on an existing quarantine that gets extended, or
        whether they are no longer eligible for quarantine, will be checked when the start_date
        is reached.

        Args:
            inds (int): indices of who to quarantine, specified by check_quar()
            start_date (int): day to begin quarantine (defaults to the current day, `sim.t`)
            period (int): quarantine duration (defaults to ``pars['quar_period']``)
        '''

        start_date = self.t if start_date is None else int(start_date)
        period = self.pars['quar_period'] if period is None else int(period)
        for ind in inds:
            self._pending_quarantine[start_date].append((ind, start_date + period))
        return


    #%% Analysis methods

    def plot(self, *args, **kwargs):
        '''
        Plot statistics of the population -- age distribution, numbers of contacts,
        and overall weight of contacts (number of contacts multiplied by beta per
        layer).

        Args:
            bins      (arr)   : age bins to use (default, 0-100 in one-year bins)
            width     (float) : bar width
            font_size (float) : size of font
            alpha     (float) : transparency of the plots
            fig_args  (dict)  : passed to pl.figure()
            axis_args (dict)  : passed to pl.subplots_adjust()
            plot_args (dict)  : passed to pl.plot()
            do_show   (bool)  : whether to show the plot
            fig       (fig)   : handle of existing figure to plot into
        '''
        fig = cvplt.plot_people(people=self, *args, **kwargs)
        return fig


    def story(self, uid, *args):
        '''
        Print out a short history of events in the life of the specified individual.

        Args:
            uid (int/list): the person or people whose story is being regaled
            args (list): these people will tell their stories too

        **Example**::

            sim = cv.Sim(pop_type='hybrid', verbose=0)
            sim.run()
            sim.people.story(12)
            sim.people.story(795)
        '''

        def label_lkey(lkey):
            ''' Friendly name for common layer keys '''
            if lkey.lower() == 'a':
                llabel = 'default contact'
            if lkey.lower() == 'h':
                llabel = 'household'
            elif lkey.lower() == 's':
                llabel = 'school'
            elif lkey.lower() == 'w':
                llabel = 'workplace'
            elif lkey.lower() == 'c':
                llabel = 'community'
            else:
                llabel = f'"{lkey}"'
            return llabel

        uids = sc.promotetolist(uid)
        uids.extend(args)

        for uid in uids:

            p = self[uid]
            sex = 'female' if p.sex == 0 else 'male'

            intro = f'\nThis is the story of {uid}, a {p.age:.0f} year old {sex}'

            if not p.susceptible:
                if np.isnan(p.date_symptomatic):
                    print(f'{intro}, who had asymptomatic COVID.')
                else:
                    print(f'{intro}, who had symptomatic COVID.')
            else:
                print(f'{intro}, who did not contract COVID.')

            total_contacts = 0
            no_contacts = []
            for lkey in p.contacts.keys():
                llabel = label_lkey(lkey)
                n_contacts = len(p.contacts[lkey])
                total_contacts += n_contacts
                if n_contacts:
                    print(f'{uid} is connected to {n_contacts} people in the {llabel} layer')
                else:
                    no_contacts.append(llabel)
            if len(no_contacts):
                nc_string = ', '.join(no_contacts)
                print(f'{uid} has no contacts in the {nc_string} layer(s)')
            print(f'{uid} has {total_contacts} contacts in total')

            events = []

            dates = {
                'date_critical'       : 'became critically ill and needed ICU care',
                'date_dead'           : 'died ☹',
                'date_diagnosed'      : 'was diagnosed with COVID',
                'date_end_quarantine' : 'ended quarantine',
                'date_infectious'     : 'became infectious',
                'date_known_contact'  : 'was notified they may have been exposed to COVID',
                'date_pos_test'       : 'recieved their positive test result',
                'date_quarantined'    : 'entered quarantine',
                'date_recovered'      : 'recovered',
                'date_severe'         : 'developed severe symptoms and needed hospitalization',
                'date_symptomatic'    : 'became symptomatic',
                'date_tested'         : 'was tested for COVID',
                "date_end_diagnosis"  : "no longer isolating"
            }

            for attribute, message in dates.items():
                date = getattr(p,attribute)
                if not np.isnan(date):
                    events.append((date, message))

            for infection in self.infection_log:
                lkey = infection['layer']
                llabel = label_lkey(lkey)
                if infection['target'] == uid:
                    if lkey:
                        events.append((infection['date'], f'was infected with COVID by {infection["source"]} via the {llabel} layer'))
                    else:
                        events.append((infection['date'], 'was infected with COVID as a seed infection'))

                if infection['source'] == uid:
                    x = len([a for a in self.infection_log if a['source'] == infection['target']])
                    events.append((infection['date'],f'gave COVID to {infection["target"]} via the {llabel} layer ({x} secondary infections)'))

            if len(events):
                for day, event in sorted(events, key=lambda x: x[0]):
                    print(f'On day {day:.0f}, {uid} {event}')
            else:
                print(f'Nothing happened to {uid} during the simulation.')
        return
