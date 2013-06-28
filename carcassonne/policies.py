# Imports {{{
from copy import copy
from numpy.linalg import norm
from .utils import O
# }}}

# Logging {{{
import logging
log = logging.getLogger(__name__)
# }}}

# Base classes {{{
class Policy: # {{{
    def createBindingToSystem(self,system):
        return self.Proxy(self,system)
# }}}
class Proxy: # {{{
    def __init__(self,forward,system): # {{{
        self.forward = forward
        self.system = system
    # }}}
    def __getattr__(self,name): # {{{
        return getattr(self.forward,name)
    # }}}
# }}}
# }}}

# Simple proxies {{{
class ApplyProxy(Proxy): # {{{
    def apply(self):
        return type(self.forward).apply(self)
# }}}
class ConvergedProxy(Proxy): # {{{
    def converged(self):
        return type(self.forward).converged(self)
# }}}
class ResetProxy(Proxy): # {{{
    def reset(self):
        return type(self.forward).reset(self)
# }}}
class UpdateProxy(Proxy): # {{{
    def update(self):
        return type(self.forward).update(self)
# }}}
# }}}

# Bandwidth policies {{{
class BandwidthIncreasePolicy(Policy): # {{{
    Proxy = ApplyProxy
# }}}
class AlternatingDirectionsIncrementBandwidthIncreasePolicy(BandwidthIncreasePolicy): # {{{
    def __init__(self,directions):
        self.directions = directions
        self.direction = 0
    def apply(self):
        self.system.increaseBandwidth(self.direction,by=self.increment,do_as_much_as_possible=True)
        self.direction = O(self.direction)
# }}}
class AllDirectionsIncrementBandwidthIncreasePolicy(BandwidthIncreasePolicy): # {{{
    def __init__(self,increment=1):
        self.increment = increment
    def apply(self):
        log.info("Increasing bandwidth in all directions by " + str(self.increment))
        for direction in (0,1):
            self.system.increaseBandwidth(direction,by=self.increment,do_as_much_as_possible=True)
# }}}
class OneDirectionIncrementBandwidthIncreasePolicy(BandwidthIncreasePolicy): # {{{
    def __init__(self,direction,increment=1):
        self.direction = direction
        self.increment = increment
    def apply(self):
        log.info("Increasing bandwidth in direction " + str(self.direction) + " by " + str(self.increment))
        self.system.increaseBandwidth(self.direction,by=self.increment,do_as_much_as_possible=True)
# }}}
# }}}

# Compression policies {{{
class CompressionPolicy(Policy): # {{{
    Proxy = ApplyProxy
# }}}
class ConstantStateCompressionPolicy(CompressionPolicy): # {{{
    def __init__(self,new_dimension):
        self.new_dimension = new_dimension
    def apply(self):
        log.debug("Compressing to " + str(self.new_dimension))
        for corner_id in range(4):
            for direction in range(2):
                self.system.compressCornerStateTowards(corner_id,direction,self.new_dimension)
# }}}
# }}}

# Contraction policies {{{
class ContractionPolicy(Policy): # {{{
    class Proxy(ApplyProxy,ResetProxy):
        pass
# }}}
class RepeatPatternContractionPolicy(ContractionPolicy): # {{{
    def __init__(self,directions):
        self.directions = directions
        self.reset()
    def apply(self):
        try:
            direction = next(self.iteration)
        except StopIteration:
            self.iteration = iter(self.directions)
            try:
                direction = next(self.iteration)
            except StopIteration:
                raise ValueError("An empty sequence of contraction directions was provided! ({})".format(self.directions))
        log.debug("Contracting towards direction {}".format(direction))
        self.system.contractTowards(direction)
    def reset(self):
        self.iteration = iter([])
# }}}
# }}}

# Convergence policies {{{
class ConvergencePolicy(Policy): # {{{
    class Proxy(ConvergedProxy,ResetProxy,UpdateProxy):
        pass
# }}}
class PeriodicyThresholdConvergencePolicy(ConvergencePolicy): # {{{
    def __init__(self,threshold,*directions):
        self.threshold = threshold
        if not directions:
            raise ValueError("at least one direction must be specified")
        self.directions = directions
    def converged(self):
        state_center_data = self.system.state_center_data
        difference = 0
        for direction in self.directions:
            normalized_data = state_center_data.normalizeAxis(direction)[0]
            denormalizer = state_center_data.normalizeAxis(O(direction))[-1]
            difference += (state_center_data-normalized_data.absorbMatrixAt(direction,denormalizer)).norm()
        return difference < self.threshold
    def reset(self):
        pass
    def update(self):
        pass
# }}}
class RelativeEstimatedOneSiteExpectationDifferenceThresholdConvergencePolicy(ConvergencePolicy): # {{{
    def __init__(self,direction,threshold):
        self.direction = direction
        self.threshold = threshold
        self.last = None
        self.current = None
    def converged(self):
        last = self.last
        current = self.current
        print("est =",current)
        return last is not None and current is not None and (abs(current+last) < 1e-15 or abs(current-last)/abs(current+last)*2 < self.threshold)
    def reset(self):
        self.last = None
        self.current = None
    def update(self):
        self.last = self.current
        self.system = copy(self.system)
        exp1 = self.system.computeExpectation()
        self.system.contractTowards(self.direction)
        exp2 = self.system.computeExpectation()
        self.current = exp2-exp1
# }}}
class RelativeExpectationDifferenceDifferenceThresholdConvergencePolicy(ConvergencePolicy): # {{{
    def __init__(self,threshold):
        self.threshold = threshold
        self.last_value = None
        self.last_difference = None
        self.current_value = None
        self.current_difference = None
    def converged(self):
        last = self.last_difference
        current = self.current_difference
        print("diff =",current)
        if last is not None and current is not None:
            absolute_difference = abs(current-last)
            relative_difference = absolute_difference/abs(current+last)*2
            return absolute_difference < 1e-15 or relative_difference < self.threshold
    def reset(self):
        self.last_value = None
        self.last_difference = None
        self.current_value = None
        self.current_difference = None
    def update(self):
        self.last_value = self.current_value
        self.last_difference = self.current_difference
        self.current_value = self.system.computeExpectation()
        if self.last_value is not None:
            self.current_difference = self.current_value-self.last_value
# }}}
class RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(ConvergencePolicy): # {{{
    def __init__(self,threshold):
        self.threshold = threshold
        self.last = None
        self.current = None
    def converged(self):
        last = self.last
        current = self.current
        if last is not None and current is not None:
            if current-last > self.threshold:
                log.info("Current expectation ({}) is greater than last expectation ({})!".format(current,last))
            absolute_difference = abs(current-last)
            relative_difference = abs(current-last)/abs(current+last)*2
            log.debug("Expectations: current = {};  last = {};  absolute difference = {};  relative difference = {}".format(current,last,absolute_difference,relative_difference))
            return (absolute_difference < 1e-15 or relative_difference < self.threshold)
    def reset(self):
        self.last = None
        self.current = None
    def update(self):
        expectation = self.system.computeOneSiteExpectation()
        self.last = self.current
        self.current = expectation
# }}}
class RelativeStateDifferenceThresholdConvergencePolicy(ConvergencePolicy): # {{{
    def __init__(self,threshold):
        self.threshold = threshold
        self.last = None
        self.current = None
    def converged(self):
        last = self.last
        current = self.current
        print("state =",current)
        if last is not None and current is not None:
            magnitude = norm(current+last)
            relative_difference = norm(current-last)/magnitude*2
            log.debug("site relative difference = {} (magnitude = {})".format(relative_difference,magnitude))
            return magnitude < 1e-15 or relative_difference < self.threshold
        else:
            return False
        return last is not None and current is not None and ()
    def reset(self):
        self.last = None
        self.current = None
    def update(self):
        self.last = self.current
        self.current = self.system.state_center_data.toArray()
# }}}
# }}}

# Hook Policy {{{
class HookPolicy(Policy):
    Proxy = ApplyProxy
    def __init__(self,callback):
        self.callback = callback
    def apply(self):
        return self.callback(self.system)
# }}}

# Exports {{{
__all__ = [
    "Policy",
    "Proxy",

    "ApplyProxy",
    "ConvergedProxy",
    "ResetProxy",
    "UpdateProxy",

    "BandwidthIncreasePolicy",
    "AllDirectionsIncrementBandwidthIncreasePolicy",
    "AlternatingDirectionsIncrementBandwidthIncreasePolicy",
    "OneDirectionIncrementBandwidthIncreasePolicy",

    "CompressionPolicy",
    "ConstantStateCompressionPolicy",

    "ContractionPolicy",
    "RepeatPatternContractionPolicy",

    "ConvergencePolicy",
    "PeriodicyThresholdConvergencePolicy",
    "RelativeEstimatedOneSiteExpectationDifferenceThresholdConvergencePolicy",
    "RelativeExpectationDifferenceDifferenceThresholdConvergencePolicy",
    "RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy",
    "RelativeStateDifferenceThresholdConvergencePolicy",

    "HookPolicy",
]
