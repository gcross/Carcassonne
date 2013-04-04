# Imports {{{
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
            "post-optimization hook",
            "state compression",
            "sweep convergence",
        ]}
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
        self.minimizeExpectation()
        self._updatePolicy("sweep convergence")
        while not self._hasConverged("sweep convergence"):
            self._applyPolicy("contraction")
            self._applyPolicy("state compression",optional=True)
            self._applyPolicy("operator compression",optional=True)
            self.iteration_number_for_sweep += 1
            self.number_of_iterations += 1
            log.info("Iteration #{} of sweep #{}".format(self.iteration_number_for_sweep,sweep_number))
            try:
                self.minimizeExpectation()
                self._applyPolicy("post-optimization hook",optional=True)
                self._updatePolicy("sweep convergence")
            except RelaxFailed:
                pass
    # }}}
  # }}}
  # Protected instance methods {{{
    def _increaseBandwidth(self,axes,by=None,to=None,do_as_much_as_possible=False): # {{{
        state_center_data = self.state_center_data
        ndim = state_center_data.ndim
        old_dimension = state_center_data.shape[axes[0]]
        new_dimension = \
            computeNewDimension(
                old_dimension,
                by=by,
                to=to,
            )
        if new_dimension == old_dimension:
            return
        if new_dimension > 2*old_dimension:
            if do_as_much_as_possible:
                new_dimension = 2*old_dimension
            else:
                raise ValueError("New dimension must be less than twice the old dimension ({} > 2*{}).".format(new_dimension,old_dimension))
        increment = new_dimension-old_dimension
        extra_state_center_data = state_center_data.reverseLastAxis()
        self.setStateCenter(
            state_center_data.increaseDimensionsAndFillWithZeros(*((axis,new_dimension) for axis in axes))
        )
        if increment == old_dimension:
            for axis in axes:
                self.contractTowards(
                    O(axis) if ndim == 5 else 1-axis,
                    state_center_data.directSumWith(
                        extra_state_center_data,
                        *dropAt(range(ndim),axis)
                    ),
                )
        else:
            for axis in axes:
                compressor, _ = \
                    computeCompressorForMatrixTimesItsDagger(
                        old_dimension,
                        increment,
                        extra_state_center_data.fold(axis).transpose().toArray()
                    )
                self.contractTowards(
                    O(axis) if ndim == 5 else 1-axis,
                    state_center_data.directSumWith(
                        extra_state_center_data.absorbMatrixAt(axis,NDArrayData(compressor)),
                        *dropAt(range(ndim),axis)
                    ),
                )
        self.just_increased_bandwidth = True
    # }}}
  # }}}
# }}}
