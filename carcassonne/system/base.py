# Imports {{{
from ..policies import OptionalPolicyField, PolicyField
from ..utils import RelaxFailed
# }}}

class BaseSystem: # {{{
  # Instance methods {{{
    def runUntilConverged(self): # {{{
        self.number_of_sweeps = 1
        self.number_of_iterations = 1
        self.sweepUntilConverged()
        self.run_convergence_policy.update()
        while not self.run_convergence_policy.converged():
            self.increase_bandwidth_policy.apply()
            self.sweepUntilConverged()
            self.run_convergence_policy.update()
    # }}}
    def sweepUntilConverged(self): # {{{
        self.number_of_sweeps += 1
        self.sweep_convergence_policy.reset()
        self.contraction_policy.reset()
        self.minimizeExpectation()
        self.sweep_convergence_policy.update()
        while not self.sweep_convergence_policy.converged():
            self.number_of_iterations += 1
            self.contraction_policy.apply()
            self.state_compression_policy.apply()
            self.operator_compression_policy.apply()
            try:
                self.minimizeExpectation()
                self.sweep_convergence_policy.update()
            except RelaxFailed:
                pass
    # }}}
  # }}}
  # Policy fields {{{
    state_compression_policy = OptionalPolicyField("state_compression_policy")
    operator_compression_policy = OptionalPolicyField("operator_compression_policy")
    sweep_convergence_policy = PolicyField("sweep_convergence_policy")
    run_convergence_policy = PolicyField("run_convergence_policy")
    increase_bandwidth_policy = PolicyField("increase_bandwidth_policy")
    contraction_policy = PolicyField("contraction_policy")
  # }}}
# }}}
