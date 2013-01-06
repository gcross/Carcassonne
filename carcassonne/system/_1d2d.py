# Imports {{{
from copy import copy
from numpy import complex128, ones, zeros
from numpy.linalg import norm

from . import _1d, _2d
from .base import BaseSystem
from ..data import NDArrayData
from ..sparse import Identity, Complete, TwoSiteOperator 
from ..utils import Pauli, buildProductTensor, buildTensor
# }}}

# Classes {{{
class System(BaseSystem): # {{{
    @classmethod
    def new(cls,rotation,O=None,OO=None): # {{{
        if OO is None and O is None:
            raise ValueError("either O or OO must not be None")
        if OO is None:
            OO = (0*O,)*2
        if O is None:
            O = 0*OO[0]
        _1d_system = _1d.System(
            [1,0,0],
            [0,0,1],
            buildTensor((3,3,2,2),{
                (0,0): Pauli.I,
                (0,2): O,
                (0,1): OO[0],
                (1,2): OO[1],
                (2,2): Pauli.I,
            }),
            ones((1,1,2)),
        )

        if O is not None:
            O = NDArrayData(O)
        if OO is not None:
            OO = NDArrayData(OO)

        if rotation == 0:
            _2d_system = _2d.System.newTrivialWithSparseOperator(O=O,OO_LR=OO)
        elif rotation == 1:
            _2d_system = _2d.System.newTrivialWithSparseOperator(O=O,OO_UD=OO)
        else:
            raise ValueError("rotation must be 0 or 1, not {}".format(rotation))

        return cls(rotation,_1d_system,_2d_system)
    # }}}
    def __init__(self,rotation,_1d,_2d): # {{{
        self.rotation = rotation
        self._1d = _1d
        self._2d = _2d
    # }}}
    def __copy__(self): # {{{
        return System(self.rotation,copy(self._1d),copy(self._2d))
    # }}}
    def check(self,prefix): # {{{
        self.checkEnvironments(prefix)
        self.checkStates(prefix)
        self.checkExpectation(prefix)
    # }}}
    def checkEnvironments(self,prefix): # {{{
        bandwidth = self._1d.center_state.shape[0]
        slot_of = { 
            Identity(): 0,
            TwoSiteOperator(2): 1,
            Complete(): 2,
        }

        _1d_right_environment = self._1d.right_environment.toArray()
        _2d_right_environment = zeros((3,) + (bandwidth,)*2,dtype=complex128)
        for tag, value in self._2d.sides[0].items():
            _2d_right_environment[slot_of[tag]] = value.toArray().reshape(bandwidth,bandwidth)
        if norm(_1d_right_environment-_2d_right_environment) > 1e-7:
            raise Exception(prefix + ": for the right environment, norm(_1d-_2d)={} > 1e-7".format(norm(_1d_right_environment-_2d_right_environment)))

        _1d_left_environment = self._1d.left_environment.toArray()
        _2d_left_environment = zeros((3,) + (bandwidth,)*2,dtype=complex128)
        for tag, value in self._2d.sides[2].items():
            _2d_left_environment[2-slot_of[tag]] = value.toArray().reshape(bandwidth,bandwidth)
        if norm(_1d_left_environment-_2d_left_environment) > 1e-7:
            raise Exception(prefix + ": for the left environment, norm(_1d-_2d)={} > 1e-7".format(norm(_1d_left_environment-_2d_left_environment)))
    # }}}
    def checkExpectation(self,prefix): # {{{
        _1d_exp = self._1d.computeExpectation()
        _2d_exp = self._2d.computeExpectation()
        if abs(_1d_exp-_2d_exp) > 1e-7:
            raise Exception(prefix + ": for the expectation, abs(1D@{} - 2D@{})={}>1e-7".format(_1d_exp,_2d_exp,abs(_1d_exp-_2d_exp)))
    # }}}
    def checkStates(self,prefix): # {{{
        _1d = self._1d.getCenterStateAsArray()
        _2d = self._2d.getCenterStateAsArray().reshape(_1d.shape)
        if norm(_1d-_2d) > 1e-7:
            _1d = copy(_1d)
            _1d[abs(_1d)<1e-10] = 0
            print(_1d)
            _2d = copy(_2d)
            _2d[abs(_2d)<1e-10] = 0
            print(_2d)
            raise Exception(prefix + ": for the center state, norm(_1d-_2d)={} > 1e-7".format(norm(_1d-_2d)))
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
        return _1d
    # }}}
    def contractTowards(self,direction): # {{{
        print("contracting towards ",direction,2*direction+self.rotation)
        self.check("before contraction, ")
        self._1d.contractTowards(direction)
        self._2d.contractTowards(2*direction+self.rotation)
        self.check("after contraction, ")
    # }}}
    def getCenterStateAsArray(self): # {{{
        self.check("while getting center state as array")
        return self._1d.getCenterStateAsArray()
    # }}}
    def increaseBandwidth(self,direction,by=None,to=None,do_as_much_as_possible=False): # {{{
        self.check("before increasing bandwidth")
        self._1d.increaseBandwidth(direction,by,to,do_as_much_as_possible)
        self._2d.increaseBandwidth(direction,by,to,do_as_much_as_possible)
        self.check("after increasing bandwidth")
    # }}}
    def minimizeExpectation(self): # {{{
        self.check("before minimizing, ")
        self._1d.minimizeExpectation()
        self._2d.minimizeExpectation()
        self.check("after minimizing, ")
    # }}}
# }}}
# }}}
