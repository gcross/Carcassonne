# Imports {{{
from copy import copy

from ..data import NDArrayData
from ..utils import O, RelaxFailed, computeCompressorForMatrixTimesItsDagger, computeNewDimension, dropAt
# }}}

# Logging {{{
import logging
log = logging.getLogger(__name__)
# }}}

class BaseSystem: # {{{
  # Internal instance methods {{{
    def _applyPolicy(self,policy_name,optional=False): # {{{
        for policy in self._getPolicy(policy_name,optional):
            policy.apply()
    # }}}
    def _hasConverged(self,policy_name): # {{{
        for policy in self._getPolicy(policy_name,False):
            return policy.converged()
    # }}}
    def _getPolicy(self,policy_name,optional=False): # {{{
        policies = self._policies
        if policy_name not in policies:
            raise ValueError("No such policy name " + policy_name)
        if policies[policy_name] == None:
            if optional:
                return []
            else:
                raise ValueError("Policy " + policy_name + " has not been set.")
        else:
            return [policies[policy_name]]
    # }}}
    def _resetPolicy(self,policy_name,optional=False): # {{{
        for policy in self._getPolicy(policy_name,optional):
            policy.reset()
    # }}}
    def _updatePolicy(self,policy_name,optional=False): # {{{
        for policy in self._getPolicy(policy_name,optional):
            policy.update()
    # }}}
  # }}}
  # Public instance methods {{{
    def __init__(self): # {{{
        self._policies = {name:None for name in [
            "bandwidth increase",
            "contraction",
            "operator compression",
            "run convergence",
            "post-contraction hook",
            "pre-optimization hook",
            "post-optimization hook",
            "state compression",
            "sweep convergence",
        ]}
    # }}}
    def computeEstimatedOneSiteExpectation(self,direction=0): # {{{
        system = copy(self)
        exp1 = system.computeExpectation()
        system.contractTowards(direction)
        exp2 = system.computeExpectation()
        return exp2-exp1
    # }}}
    def runUntilConverged(self): # {{{
        log.info("Beginning run.")
        self.number_of_sweeps = 0
        self.number_of_iterations = 0
        self.sweepUntilConverged()
        self._updatePolicy("run convergence")
        while not self._hasConverged("run convergence"):
            self._applyPolicy("bandwidth increase")
            self.sweepUntilConverged()
            self._updatePolicy("run convergence")
        log.info("Finished run with {} total sweeps and {} total iterations.".format(self.number_of_sweeps,self.number_of_iterations))
    # }}}
    def setPolicy(self,policy_name,policy): # {{{
        policies = self._policies
        if policy_name not in policies:
            raise ValueError("No such policy name " + policy_name)
        if policies[policy_name] != None:
            raise ValueError("Policy " + policy_name + " has already been set.")
        policies[policy_name] = policy.createBindingToSystem(self)
    # }}}
    def sweepUntilConverged(self): # {{{
        self.number_of_sweeps += 1
        sweep_number = self.number_of_sweeps
        log.info("Starting sweep #{}".format(sweep_number))
        self._resetPolicy("contraction")
        self._resetPolicy("sweep convergence")
        self.iteration_number_for_sweep = 1
        log.info("Iteration #{} of sweep #{}".format(self.iteration_number_for_sweep,sweep_number))
        self._applyPolicy("pre-optimization hook",optional=True)
        self.minimizeExpectation()
        self._applyPolicy("post-optimization hook",optional=True)
        self._updatePolicy("sweep convergence")
        while not self._hasConverged("sweep convergence"):
            self._applyPolicy("contraction")
            self._applyPolicy("state compression",optional=True)
            self._applyPolicy("operator compression",optional=True)
            self.iteration_number_for_sweep += 1
            self.number_of_iterations += 1
            log.info("Iteration #{} of sweep #{}".format(self.iteration_number_for_sweep,sweep_number))
            try:
                self._applyPolicy("pre-optimization hook",optional=True)
                self.minimizeExpectation()
                self._applyPolicy("post-optimization hook",optional=True)
                self._updatePolicy("sweep convergence")
            except RelaxFailed:
                pass
    # }}}
  # }}}
  # Protected instance methods {{{
    def _increaseBandwidth(self,axis,by=None,to=None,do_as_much_as_possible=False,enlargeners=None): # {{{
        print("increasing in direction",axis)
        state_center_data = self.state_center_data
        ndim = state_center_data.ndim
        if ndim == 5:
            O_axis = O(axis)
        else:
            O_axis = 1-axis

        physical_dimension = state_center_data.shape[-1]
        old_dimension = state_center_data.shape[axis]
        new_dimension = \
            computeNewDimension(
                old_dimension,
                by=by,
                to=to,
            )

        if new_dimension == old_dimension:
            return
        if new_dimension > physical_dimension*old_dimension:
            if do_as_much_as_possible:
                new_dimension = physical_dimension*old_dimension
            else:
                raise ValueError("New dimension must be less than the physical dimension times the old dimension ({} > {}*{}).".format(new_dimension,physical_dimension,old_dimension))

        neighbor_0 = state_center_data.normalizeAxis(O_axis)[0]
        print("neighbor 0 @ 1 =",neighbor_0)
        neighbor_1 = state_center_data.normalizeAxis(axis)[0]
        print("neighbor 1 @ 1 =",neighbor_1)

        if enlargeners is None:
            enlargener_A, enlargener_B = state_center_data.newEnlargener(old_dimension,new_dimension)
        else:
            enlargener_A, enlargener_B = enlargeners
        print("state @ 1 =",state_center_data)
        state_center_data = state_center_data.absorbMatrixAt(axis,enlargener_A)
        neighbor_0 = neighbor_0.absorbMatrixAt(O_axis,enlargener_B)
        print("state @ 2 =",state_center_data)
        print("neighbor 0 @ 2 =",neighbor_0)
        neighbor_0, state_center_data = neighbor_0.normalizeAxisAndDenormalize(O_axis,axis,state_center_data)
        print("neighbor 0 @ 3 =",neighbor_0)
        print("state @ 3 =",state_center_data)

        neighbor_1 = neighbor_1.absorbMatrixAt(axis,enlargener_A)
        print("neighbor 1 @ 2 =",neighbor_1)
        state_center_data = state_center_data.absorbMatrixAt(O_axis,enlargener_B)
        neighbor_1, state_center_data = neighbor_1.normalizeAxisAndDenormalize(axis,O_axis,state_center_data)
        print("neighbor 1 @ 3 =",neighbor_1)
        print("state @ 3 =",state_center_data)

        self.setStateCenter(state_center_data)
        self.contractUnnormalizedTowards(axis,neighbor_0)
        self.contractUnnormalizedTowards(O_axis,neighbor_1)

        self.just_increased_bandwidth = True
        return enlargener_A, enlargener_B
    # }}}
  # }}}
# }}}
