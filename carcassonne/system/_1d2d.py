# Imports {{{
from copy import copy
from numpy import complex128, ones, sqrt, zeros
from numpy.linalg import norm

from . import _1d, _2d
from .base import BaseSystem
from ..data import NDArrayData
from ..sparse import Identity, Complete, TwoSiteOperator, makeMPO
from ..utils import Pauli, buildProductTensor, buildTensor
# }}}

# Classes {{{
class System(BaseSystem): # {{{
    @classmethod # new {{{
    def new(cls,rotation,Os=[],OOs=[],adaptive_state_threshold=False):
        operator, right_operator_boundary, right_tags, left_operator_boundary, left_tags = makeMPO(Pauli.I,Os,OOs)
        _1d_system = _1d.System(right_operator_boundary,left_operator_boundary,operator,ones((1,1,2),dtype=complex128)/sqrt(2))

        Os = map(NDArrayData,Os)
        OOs = map(NDArrayData,OOs)

        if rotation == 0:
            _2d_system = _2d.System.newTrivialWithSparseOperator(Os=Os,OO_LRs=OOs)
        elif rotation == 1:
            _2d_system = _2d.System.newTrivialWithSparseOperator(Os=Os,OO_UDs=OOs)
        else:
            raise ValueError("rotation must be 0 or 1, not {}".format(rotation))

        return cls(
            rotation,
            _1d_system,
            _2d_system,
            right_tags,
            left_tags,
            adaptive_state_threshold,
        )
    # }}}
    @classmethod # newSimple {{{
    def newSimple(cls,rotation,O=None,OO=None,adaptive_state_threshold=False):
        if O is None:
            Os = []
        else:
            Os = [O]
        if OO is None:
            OOs = []
        else:
            OOs = [OO]
        return cls.new(rotation,Os,OOs,adaptive_state_threshold)
    # }}}
    def __init__(self,rotation,_1d,_2d,right_tags,left_tags,adaptive_state_threshold): # {{{
        BaseSystem.__init__(self)
        self.rotation = rotation
        self._1d = _1d
        self._2d = _2d
        self.right_tags = right_tags
        self.left_tags = left_tags
        self.adaptive_state_threshold = adaptive_state_threshold
        self.state_threshold = 1e-5
    # }}}
    def __copy__(self): # {{{
        return System(self.rotation,copy(self._1d),copy(self._2d),self.right_tags,self.left_tags,self.adaptive_state_threshold)
    # }}}
    def check(self,prefix,threshold=1e-5): # {{{
        self.checkStates(prefix)
        self.checkEnvironments(prefix)
        self.checkNormalization(prefix)
        self.checkExpectation(prefix)
        self.checkExpectationMatrix(prefix)
    # }}}
    def checkEnvironments(self,prefix): # {{{
        bandwidth = self._1d.state_center_data.shape[0]
        _1d_left_environment = self._1d.left_environment.toArray()
        _2d_left_environment = self.convert2DLeftEnvironment().toArray()

        _1d_right_environment = self._1d.right_environment.toArray()
        _2d_right_environment = self.convert2DRightEnvironment().toArray()
        if norm(_1d_right_environment-_2d_right_environment) > 1e-7:
            raise Exception(prefix + ": for the right environment, norm(_1d-_2d)={} > 1e-7".format(norm(_1d_right_environment-_2d_right_environment)))

        if norm(_1d_left_environment-_2d_left_environment) > 1e-7:
            raise Exception(prefix + ": for the left environment, norm(_1d-_2d)={} > 1e-7".format(norm(_1d_left_environment-_2d_left_environment)))
    # }}}
    def checkExpectation(self,prefix): # {{{
        _1d_exp = self._1d.computeExpectation()
        _2d_exp = self._2d.computeExpectation()
        if abs(_1d_exp-_2d_exp) > 1e-7:
            raise Exception(prefix + ": for the expectation, abs(1D@{} - 2D@{})={} > 1e-7".format(_1d_exp,_2d_exp,abs(_1d_exp-_2d_exp)))
    # }}}
    def checkExpectationMatrix(self,prefix): # {{{
        _1d = self._1d.formExpectationMatrix().toArray()
        _2d = self._2d.formExpectationMatrix().toArray().reshape(_1d.shape)
        if norm(_1d-_2d) > 1e-7:
            raise Exception(prefix + ": for the expectation matrix, norm(1D 2D)={} > 1e-7".format(norm(_1d-_2d)))
    # }}}
    def checkNormalization(self,prefix): # {{{
        if abs(self._2d.computeNormalization()-1) > 1e-7:
            raise Exception(prefix + ": the 2D normalization is {} != 1".format(self._2d.computeNormalization()))
    # }}}
    def checkStates(self,prefix): # {{{
        _1d = copy(self._1d.state_center_data.toArray()).real
        for x in _1d.ravel():
            if abs(x) > 1e-12:
                _1d /= x
                break
        _2d = copy(self._2d.state_center_data.toArray()).reshape(_1d.shape).real
        for x in _2d.ravel():
            if abs(x) > 1e-12:
                _2d /= x
                break
        relative_norm = norm(_1d-_2d)/norm(abs(_1d)+abs(_2d))*2
        if relative_norm > self.state_threshold:
            raise Exception(prefix + ": for the center state, norm(_1d-_2d)={} > {}".format(relative_norm,self.state_threshold))
        return self._1d.state_center_data
    # }}}
    def computeExpectation(self): #{{{
        self.check("while computing expectation")
        return self._1d.computeExpectation()
    # }}}
    def computeOneSiteExpectation(self): #{{{
        _1d = self._1d.computeOneSiteExpectation()
        _2d = self._2d.computeOneSiteExpectation()
        if abs(abs(_1d)-abs(_2d)) > 1e-7:
            raise Exception("for the one-site expectation, abs({}-{})={}>1e-7".format(abs(_1d),abs(_2d),abs(abs(_1d)-abs(_2d))))
        return _2d
    # }}}
    def contractTowards(self,direction): # {{{
        self.check("before contraction, ")
        self._1d.contractTowards(direction)
        self._2d.contractTowards(2*direction+self.rotation)
        self.check("after contraction, ")
    # }}}
    def convert1DEnvironment(self,boundary,tags): # {{{
        return {tag: boundary[index].split(*((1,)*6+boundary[index].shape)) for index, tag in enumerate(tags)}
    # }}}
    def convert1DLeftEnvironment(self): # {{{
        return self.convert1DEnvironment(self._1d.left_environment,self.left_tags)
    # }}}
    def convert1DRightEnvironment(self): # {{{
        return self.convert1DEnvironment(self._1d.right_environment,self.right_tags)
    # }}}
    def convert2DLeftEnvironment(self): # {{{
        bandwidth = self._2d.state_center_data.shape[self.rotation]
        _1d_left_environment = zeros((len(self.left_tags),) + (bandwidth,)*2,dtype=complex128)
        for tag, value in self._2d.sides[self.rotation+2].items():
            _1d_left_environment[self.left_tags.index(tag)] = value.toArray().reshape(bandwidth,bandwidth)
        return NDArrayData(_1d_left_environment)
    # }}}
    def convert2DRightEnvironment(self): # {{{
        bandwidth = self._2d.state_center_data.shape[self.rotation]
        _1d_right_environment = zeros((len(self.right_tags),) + (bandwidth,)*2,dtype=complex128)
        for tag, value in self._2d.sides[self.rotation].items():
            _1d_right_environment[self.right_tags.index(tag)] = value.toArray().reshape(bandwidth,bandwidth)
        return NDArrayData(_1d_right_environment)
    # }}}
    def copy1Dto2D(self): # {{{
        b, d = self._1d.state_center_data.shape[-2:]
        if self.rotation == 0:
            self._2d.setStateCenter(NDArrayData(self._1d.state_center_data.toArray().reshape(b,1,b,1,d)))
        else:
            self._2d.setStateCenter(NDArrayData(self._1d.state_center_data.toArray().reshape(1,b,1,b,d)))
        self._2d.sides[self.rotation+2] = self.convert1DLeftEnvironment()
        self._2d.sides[self.rotation+0] = self.convert1DRightEnvironment()
    # }}}
    def copy2Dto1D(self): # {{{
        self._1d.setStateCenter(self._2d.state_center_data.join((0,1),(2,3),4))
        self._1d.left_environment = self.convert2DLeftEnvironment()
        self._1d.right_environment = self.convert2DRightEnvironment()
    # }}}
    def increaseBandwidth(self,direction,by=None,to=None,do_as_much_as_possible=False): # {{{
        self.check("before increasing bandwidth")
        # NOTE: At this time we do not directly compare the result of increaseBandwidth()
        #       because it uses a random matrix to enlarge the bandwidth dimension;
        #       fortunately both 1D and 2D use mostly the same code so this hopefully won't
        #       be a problem.
        #self._1d.increaseBandwidth(direction,by,to,do_as_much_as_possible)
        self._2d.increaseBandwidth(direction,by,to,do_as_much_as_possible)
        self.copy2Dto1D()
        self.check("after increasing bandwidth")
    # }}}
    def minimizeExpectation(self): # {{{
        self.check("before minimizing")
        self._2d.setStateCenter(NDArrayData(self._1d.state_center_data.toArray().reshape(self._2d.state_center_data.shape)))

        original = self._1d.state_center_data.toArray().ravel()
        for x in original:
            if abs(x) > 1e-12:
                original /= x
        multiplied = self._1d.formExpectationMultiplier()(self._1d.state_center_data).toArray().ravel()
        for x in multiplied:
            if abs(x) > 1e-12:
                multiplied /= x
        diff = norm(original-multiplied)/(norm(original)+norm(multiplied))*2

        if self.adaptive_state_threshold:
            self.state_threshold = 1e-5/diff
        self._1d.minimizeExpectation()
        self._2d.minimizeExpectation()
        self.check("after minimizing")
        if self.adaptive_state_threshold:
            self.state_threshold = 1e-5

        self.copy2Dto1D()
    # }}}
    state_center_data = property(lambda self: self.checkStates("when fetching state"))
# }}}
# }}}
