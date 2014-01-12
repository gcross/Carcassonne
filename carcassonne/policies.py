# Imports {{{
from numpy.linalg import norm
from .utils import O
# }}}

# Logging {{{
import logging
log = logging.getLogger(__name__)
# }}}

# Base classes {{{
class Policy: # {{{
    """A :class:`Policy` defines the behavior of some aspect of simulating a system."""
    def createBindingToSystem(self,system):
        """\
Returns the result of binding this policy to a *system*, which is a
:class:`Proxy` object that contains a reference to the system and forwards all
requests for attributes to this :class:`Policy` instance.\
"""
        return self.Proxy(self,system)
# }}}
class Proxy: # {{{
    """\
A :class:`Proxy` contains a reference to a *system* as well as a place to
*forward* requests for non-existent fields; it is created as the result of
calling :meth:`Policy.createBindingToSystem`.\
"""
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
    """An instance of :class:`ApplyProxy` has an :meth:`apply` method."""
    def apply(self):
        """Calls the `apply` method for `self.forward`'s class using this instance."""
        return type(self.forward).apply(self)
# }}}
class ConvergedProxy(Proxy): # {{{
    """An instance of :class:`ConvergedProxy` has an :meth:`converged` method."""
    def converged(self):
        """Calls the `converged` method for `self.forward`'s class using this instance."""
        return type(self.forward).converged(self)
# }}}
class ResetProxy(Proxy): # {{{
    """An instance of :class:`ResetProxy` has an :meth:`reset` method."""
    def reset(self):
        """Calls the `reset` method for `self.forward`'s class using this instance."""
        return type(self.forward).reset(self)
# }}}
class UpdateProxy(Proxy): # {{{
    """An instance of :class:`UpdatedProxy` has an :meth:`update` method."""
    def update(self):
        """Calls the `update` method for `self.forward`'s class using this instance."""
        return type(self.forward).update(self)
# }}}
# }}}

# Bandwidth policies {{{
class BandwidthIncreasePolicy(Policy): # {{{
    """A :class:`BandwidthIncreasePolicy` determines how to increase the bandwidth of a system; to use it, call the `apply` method."""
    Proxy = ApplyProxy
# }}}
class AllDirectionsIncrementBandwidthIncreasePolicy(BandwidthIncreasePolicy): # {{{
    """A :class:`BandwidthIncreasePolicy` that increases the bandwidth in both directions by *increment*."""
    def __init__(self,increment=1):
        self.increment = increment
    def apply(self):
        log.info("Increasing bandwidth in all directions by " + str(self.increment))
        for direction in (0,1):
            self.system.increaseBandwidth(direction,by=self.increment,do_as_much_as_possible=True)
# }}}
class OneDirectionIncrementBandwidthIncreasePolicy(BandwidthIncreasePolicy): # {{{
    """A :class:`BandwidthIncreasePolicy` that increases the bandwidth in *direction* by *increment*."""
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
    """A :class:`CompressionPolicy` determines how to compress the bandwidth between the corners and sides; to use it, call the `apply` method."""
    Proxy = ApplyProxy
# }}}
class ConstantStateCompressionPolicy(CompressionPolicy): # {{{
    """A :class:`CompressionPolicy` that compresses all dimensions to *new_dimension*."""
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
    """A :class:`ContractionPolicy` determines in which directions to contract the system; to perform a contraction, call the `apply` method; to reset the current direction, call the `reset` method."""
    class Proxy(ApplyProxy,ResetProxy):
        pass
# }}}
class RepeatPatternContractionPolicy(ContractionPolicy): # {{{
    """A :class:`ContractionPolicy` that rotates contracting through the directions in the order given by *directions*."""
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
    """\
A :class:`ConvergencePolicy` determines when a stage of the simulation has converged.  It has the following methods:

converged
    returns true if convergence has been reached
reset
    reset all information used to determine convergence
update
    update all information used to determine convergence given the current state of the system \
"""
    class Proxy(ConvergedProxy,ResetProxy,UpdateProxy):
        pass
# }}}
class PeriodicyThresholdConvergencePolicy(ConvergencePolicy): # {{{
    """\
A :class:`ConvergencePolicy` that determines whether the simulation has
converged based on whether the state tensor is periodic, i.e. based on whether
normalizing it in each of the *directions* and then absorbing the denormalizer
on the opposite side obtains the original state site tensor to within the
absolute norm tolerance set by *threshold*.\
"""
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
    """\
A :class:`ConvergencePolicy` that determines whether the simulation has
converged based on wheter the one-site expectation has conveged relatively to
within the tolerance specified in *threshold*.\
"""
    def __init__(self,threshold,direction=0):
        self.direction = direction
        self.threshold = threshold
        self.last = None
        self.current = None
    def converged(self):
        last = self.last
        current = self.current
        if last is not None and current is not None:
            magnitude = abs(current+last)
            absolute_difference = abs(current-last)
            relative_difference = absolute_difference/abs(current+last)*2
            log.debug("current estimated one site expectation = {}, absolute difference = {}, relative difference = {}, magnitude = {}".format(current,absolute_difference,relative_difference,magnitude))
            return magnitude < 1e-15 or relative_difference < self.threshold
    def reset(self):
        self.last = None
        self.current = None
    def update(self):
        self.last = self.current
        self.current = self.system.computeEstimatedOneSiteExpectation(self.direction)
# }}}
class RelativeExpectationDifferenceDifferenceThresholdConvergencePolicy(ConvergencePolicy): # {{{
    """\
A :class:`ConvergencePolicy` that determines whether the simulation has
converged based on wheter the system's expectation has conveged relatively to
within the tolerance specified in *threshold*.\
"""
    def __init__(self,threshold):
        self.threshold = threshold
        self.last_value = None
        self.last_difference = None
        self.current_value = None
        self.current_difference = None
    def converged(self):
        last = self.last_difference
        current = self.current_difference
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
    """\
A :class:`ConvergencePolicy` that determines whether the simulation has
converged based on wheter the one-site expectation has conveged relatively to
within the tolerance specified in *threshold*.\
"""
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
    """\
A :class:`ConvergencePolicy` that determines whether the simulation has
converged based on wheter the state tenspr has conveged relatively to within
the norm tolerance specified in *threshold*.\
"""
    def __init__(self,threshold):
        self.threshold = threshold
        self.last = None
        self.current = None
    def converged(self):
        last = self.last
        current = self.current
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
    """A :class:`HookPolicy` provides a *callback* to be called when `apply` is called."""
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
