# Imports {{{
from .utils import O
from .system import System
# }}}

# Simulators {{{
class Simulator1D: # {{{
    def __init__(self,direction,O=None,OO=None): # {{{
        if direction not in (0,1):
            raise ValueError("Direction must be either 0 for horizontal or 1 for vertial, not {}.".format(direction))
        self.direction = direction
        if direction == 0:
            self.system = System.newTrivialWithSparseOperator(O=O,OO_LR=OO)
        else:
            self.system = System.newTrivialWithSparseOperator(O=O,OO_UD=OO)
        self.bandwidth = 1
    # }}}
    def computeOneSiteExpectation(self): # {{{
        return self.system.computeCenterSiteExpectation()
    # }}}
    def runUntilConverged(self,isConverged,increaseBandwidth,makeSweepConvergenceTest): # {{{
        self.sweepUntilConverged(makeSweepConvergenceTest())
        while not isConverged(self.computeOneSiteExpectation()):
            self.bandwidth = increaseBandwidth(self.bandwidth)
            self.system.increaseBandwidth(direction=self.direction,to=self.bandwidth)
            self.sweepUntilConverged(makeSweepConvergenceTest())
    # }}}
    def sweepUntilConverged(self,isConverged): # {{{
        self.optimizeSite()
        direction = self.direction
        while not isConverged(self.computeOneSiteExpectation()):
            self.system.contractTowardsAndNormalizeCenter(direction)
            direction = O(direction)
            self.optimizeSite()
    # }}}
    def optimizeSite(self): # {{{
        self.system.minimizeExpectation()
    # }}}
# }}}

# Functions {{{
def makeConvergenceThresholdTest(threshold): # {{{
    last_value = None
    def isConverged(next_value):
        nonlocal last_value
        answer = (last_value is not None) and abs(next_value-last_value) < threshold
        last_value = next_value
        return answer
    return isConverged
# }}}
# }}}

# Exports {{{
__all__ = [
    "Simulator1D",

    "makeConvergenceThresholdTest",
]
# }}}
