# Imports {{{
from ..utils import RelaxFailed
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
            "state compression",
            "sweep convergence",
        ]}
    # }}}
    def runUntilConverged(self): # {{{
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
        iteration_number_for_sweep = 1
        log.info("Iteration #{} of sweep #{}".format(iteration_number_for_sweep,sweep_number))
        self.minimizeExpectation()
        self._updatePolicy("sweep convergence")
        while not self._hasConverged("sweep convergence"):
            iteration_number_for_sweep += 1
            log.info("Iteration #{} of sweep #{}".format(iteration_number_for_sweep,sweep_number))
            self._applyPolicy("contraction")
            self._applyPolicy("state compression",optional=True)
            self._applyPolicy("operator compression",optional=True)
            try:
                self.minimizeExpectation()
                self._updatePolicy("sweep convergence")
            except RelaxFailed:
                pass
        self.number_of_iterations += iteration_number_for_sweep
        
    # }}}
  # }}}
# }}}
