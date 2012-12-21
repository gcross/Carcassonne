# Imports {{{
from ..policies import PolicyField
# }}}

class BaseSystem: # {{{
  # Instance methods {{{
    def runUntilConverged(self): # {{{
        self.sweepUntilConverged()
        self.run_convergence_policy.update()
        while not self.run_convergence_policy.converged():
            self.increase_bandwidth_policy.apply()
            self.sweepUntilConverged()
            self.run_convergence_policy.update()
    # }}}
    def sweepUntilConverged(self): # {{{
        self.sweep_convergence_policy.reset()
        self.contraction_policy.reset()
        self.minimizeExpectation()
        self.sweep_convergence_policy.update()
        while not self.sweep_convergence_policy.converged():
            self.contraction_policy.apply()
            pre = self.computeExpectation()
            self.minimizeExpectation()
            post = self.computeExpectation()
            assert post < pre + 1e-7
            self.sweep_convergence_policy.update()
    # }}}
  # }}}
  # Policy fields {{{
    sweep_convergence_policy = PolicyField("sweep_convergence_policy")
    run_convergence_policy = PolicyField("run_convergence_policy")
    increase_bandwidth_policy = PolicyField("increase_bandwidth_policy")
    contraction_policy = PolicyField("contraction_policy")
  # }}}
# }}}
