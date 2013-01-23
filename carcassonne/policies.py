# Imports {{{
from copy import copy
from numpy.linalg import norm
# }}}

# Base classes {{{
class Policy: # {{{
    def __call__(self,system):
        return self.BoundPolicy(system,self)
# }}}
class BoundPolicyBase: # {{{
    def __init__(self,system,parent):
        self.system = system
        self.parent = parent
# }}}
class BoundBandwidthPolicy(BoundPolicyBase): # {{{
    def reset(self):
        pass
# }}}
class BoundContractionPolicy(BoundPolicyBase): # {{{
    def reset(self):
        pass
# }}}
# }}}

# Bandwidth policies {{{
class AlternatingDirectionsIncrementBandwidthPolicy(Policy): # {{{
    def __init__(self,directions):
        self.directions = directions
    class BoundPolicy(BoundBandwidthPolicy):
        def __init__(self,system,parent):
            BoundBandwidthPolicy.__init__(self,system,parent)
            self.direction = 0
        def apply(self):
            self.system.increaseBandwidth(self.direction,by=self.parent.increment,do_as_much_as_possible=True)
            self.direction = O(self.direction)
# }}}
class AllDirectionsIncrementBandwidthPolicy(Policy): # {{{
    def __init__(self,increment=1):
        self.increment = increment
    class BoundPolicy(BoundBandwidthPolicy):
        def apply(self):
            for direction in (0,1):
                self.system.increaseBandwidth(direction,by=self.parent.increment,do_as_much_as_possible=True)
# }}}
class OneDirectionIncrementBandwidthPolicy(Policy): # {{{
    def __init__(self,direction,increment=1):
        self.direction = direction
        self.increment = increment
    class BoundPolicy(BoundBandwidthPolicy):
        def apply(self):
            self.system.increaseBandwidth(self.parent.direction,by=self.parent.increment,do_as_much_as_possible=True)
# }}}
# }}}

# Compression policies {{{
class ConstantStateCompressionPolicy(Policy): # {{{
    def __init__(self,new_dimension):
        self.new_dimension = new_dimension
    class BoundPolicy(BoundBandwidthPolicy):
        def apply(self):
            for corner_id in range(4):
                print("BEFORE[{}]: {}".format(corner_id,self.system.corners[corner_id]))
                for direction in range(2):
                    self.system.compressCornerStateTowards(corner_id,direction,self.parent.new_dimension)
                    print("AFTER[{}]:{}: {}".format(corner_id,direction,self.system.corners[corner_id]))
# }}}
# }}}

# Contraction policies {{{
class RepeatPatternContractionPolicy(Policy): # {{{
    def __init__(self,directions):
        self.directions = directions
    class BoundPolicy(BoundContractionPolicy):
        def __init__(self,system,parent):
            BoundContractionPolicy.__init__(self,system,parent)
            self.iteration = iter([])
        def apply(self):
            try:
                direction = next(self.iteration)
            except StopIteration:
                self.iteration = iter(self.parent.directions)
                try:
                    direction = next(self.iteration)
                except StopIteration:
                    raise ValueError("An empty sequence of contraction directions was provided! ({})".format(self.parent.directions))
            self.system.contractTowards(direction)
# }}}
# }}}

# Convergence policies {{{
class RelativeEstimatedOneSiteExpectationDifferenceThresholdConvergencePolicy(Policy): # {{{
    def __init__(self,direction,threshold):
        self.direction = direction
        self.threshold = threshold
    class BoundPolicy(BoundPolicyBase):
        def __init__(self,system,parent):
            BoundPolicyBase.__init__(self,system,parent)
            self.last = None
            self.current = None
        def converged(self):
            last = self.last
            current = self.current
            return last is not None and current is not None and (abs(current+last) < 1e-15 or abs(current-last)/abs(current+last)*2 < self.parent.threshold)
        def reset(self):
            self.last = None
            self.current = None
        def update(self):
            self.last = self.current
            system = copy(self.system)
            exp1 = system.computeExpectation()
            system.contractTowards(self.parent.direction)
            exp2 = system.computeExpectation()
            self.current = exp2-exp1
# }}}
class RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(Policy): # {{{
    def __init__(self,threshold):
        self.threshold = threshold
    class BoundPolicy(BoundPolicyBase):
        def __init__(self,system,parent):
            BoundPolicyBase.__init__(self,system,parent)
            self.last = None
            self.current = None
        def converged(self):
            last = self.last
            current = self.current
            return last is not None and current is not None and (abs(current+last) < 1e-15 or abs(current-last)/abs(current+last)*2 < self.parent.threshold)
        def reset(self):
            self.last = None
            self.current = None
        def update(self):
            self.last = self.current
            self.current = self.system.computeOneSiteExpectation()
# }}}
class RelativeStateDifferenceThresholdConvergencePolicy(Policy): # {{{
    def __init__(self,threshold):
        self.threshold = threshold
    class BoundPolicy(BoundPolicyBase):
        def __init__(self,system,parent):
            BoundPolicyBase.__init__(self,system,parent)
            self.last = None
            self.current = None
        def converged(self):
            last = self.last
            current = self.current
            #if last is not None and current is not None:
            #    print(norm(current-last)/norm(current+last)*2)
            return last is not None and current is not None and (norm(current+last) < 1e-15 or norm(current-last)/norm(current+last)*2 < self.parent.threshold)
        def reset(self):
            self.last = None
            self.current = None
        def update(self):
            self.last = self.current
            self.current = self.system.getCenterStateAsArray()
# }}}
# }}}

# Policy field descriptor class {{{
class PolicyField:
    def __init__(self,name):
        self.name = name
    def __get__(self,instance,owner):
        try:
            return self.policy
        except AttributeError:
            raise AttirbuteError("Policy '%s' has not been set.".format(self.name))
    def __set__(self,instance,unbound_policy):
        self.policy = unbound_policy(instance)
# }}}

# OptionalPolicy field descriptor class {{{
class OptionalPolicyField(PolicyField):
    def __get__(self,instance,owner):
        try:
            return self.policy
        except AttributeError:
            class Dummy:
                @staticmethod
                def __getattr__(_):
                    return lambda: None
            return Dummy()
# }}}

# Exports {{{
__all__ = [
    "Policy",
    "BoundPolicyBase",
    "BoundBandwidthPolicy",
    "BoundContractionPolicy",

    "AllDirectionsIncrementBandwidthPolicy",
    "AlternatingDirectionsIncrementBandwidthPolicy",
    "OneDirectionIncrementBandwidthPolicy",

    "ConstantStateCompressionPolicy",

    "RepeatPatternContractionPolicy",

    "RelativeEstimatedOneSiteExpectationDifferenceThresholdConvergencePolicy",
    "RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy",
    "RelativeStateDifferenceThresholdConvergencePolicy",

    "PolicyField",
    "OptionalPolicyField",
]
