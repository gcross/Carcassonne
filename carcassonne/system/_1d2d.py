# Imports {{{
from copy import copy
from numpy import complex128, ones, sqrt, zeros
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
                (0,1): OO[1],
                (1,2): OO[0],
                (2,2): Pauli.I,
            }),
            ones((1,1,2),dtype=complex128)/sqrt(2),
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
        self.checkStates(prefix)
        self.checkEnvironments(prefix)
        self.checkNormalization(prefix)
        self.checkExpectation(prefix)
        self.checkExpectationMatrix(prefix)
    # }}}
    def checkEnvironments(self,prefix): # {{{
        bandwidth = self._1d.center_state.shape[0]
        _1d_left_environment = self._1d.left_environment.toArray()
        _2d_left_environment = self.convert2DLeftEnvironment().toArray()

        _1d_right_environment = self._1d.right_environment.toArray()
        _2d_right_environment = self.convert2DRightEnvironment().toArray()
        if norm(_1d_right_environment-_2d_right_environment) > 1e-7:
            #print(_1d_left_environment)
            #print(_2d_left_environment)
            #print(_1d_right_environment)
            #print(_2d_right_environment)
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
        _1d = copy(self._1d.getCenterStateAsArray()).real
        _2d = copy(self._2d.getCenterStateAsArray().reshape(_1d.shape)).real
        if norm(_1d-_2d) > 1e-5:
            #_1d[abs(_1d)<1e-7] = 0
            #_2d[abs(_2d)<1e-7] = 0
            #print(_1d)
            #print(_2d)
            raise Exception(prefix + ": for the center state, norm(_1d-_2d)={} > 1e-5".format(norm(_1d-_2d)))
    # }}}
    def computeExpectation(self): #{{{
        self.check("while computing expectation")
        return self._1d.computeExpectation()
    # }}}
    def computeOneSiteExpectation(self): #{{{
        #_1d = self._1d.computeOneSiteExpectation()
        print("limiting exp = {:.15}".format(self._1d.computeOneSiteExpectation().real))

        _1d = 0
        _1d_system = copy(self._1d)

        left_environment = 0*_1d_system.left_environment.toArray()
        left_environment[-1] = self._1d.left_environment.toArray()[-1]
        _1d_system.left_environment = NDArrayData(left_environment)
        del left_environment

        right_environment = 0*_1d_system.right_environment.toArray()
        right_environment[0] = self._1d.right_environment.toArray()[0]
        _1d_system.right_environment = NDArrayData(right_environment)
        del right_environment

        center_operator = 0*_1d_system.center_operator.toArray()
        center_operator[0,2] = self._1d.center_operator.toArray()[0,2]
        _1d_system.center_operator = NDArrayData(center_operator)
        _1d += _1d_system.computeExpectation()
        print("O EXP =",_1d_system.computeExpectation())
        del center_operator

        center_operator = 0*_1d_system.center_operator.toArray()
        center_operator[0,1] = self._1d.center_operator.toArray()[0,1]
        _1d_system.center_operator = NDArrayData(center_operator)
        del center_operator
        _1d_system.contractRight()
        _1d_system.center_operator = self._1d.center_operator
        #_1d_system.minimizeExpectation()
        center_operator = 0*_1d_system.center_operator.toArray()
        center_operator[1,2] = self._1d.center_operator.toArray()[1,2]
        _1d_system.center_operator = NDArrayData(center_operator)
        del center_operator
        _1d += _1d_system.computeExpectation()
        print("OO EXP =",_1d_system.computeExpectation())

        del _1d_system


        _2d = self._2d.computeOneSiteExpectation()
        if abs(abs(_1d)-abs(_2d)) > 1e-7:
            raise Exception("for the one-site expectation, abs({}-{})={}>1e-7".format(abs(_1d),abs(_2d),abs(abs(_1d)-abs(_2d))))
        return _1d
    # }}}
    def contractTowards(self,direction): # {{{
        print("contracting towards ",direction,2*direction+self.rotation)
        self.check("before contraction, ")
        #self._1d.contractTowards(direction)
        self._2d.contractTowards(2*direction+self.rotation)
        self.copy2Dto1D()
        self.check("after contraction, ")
    # }}}
    def convert2DLeftEnvironment(self): # {{{
        slot_of = { 
            Identity(): 2,
            TwoSiteOperator(2): 1,
            Complete(): 0,
        }

        bandwidth = self._2d.state_center_data.shape[self.rotation]
        _1d_left_environment = zeros((3,) + (bandwidth,)*2,dtype=complex128)
        for tag, value in self._2d.sides[self.rotation+2].items():
            _1d_left_environment[slot_of[tag]] = value.toArray().reshape(bandwidth,bandwidth)
        return NDArrayData(_1d_left_environment)
    # }}}
    def convert2DRightEnvironment(self): # {{{
        slot_of = { 
            Identity(): 0,
            TwoSiteOperator(2): 1,
            Complete(): 2,
        }

        bandwidth = self._2d.state_center_data.shape[self.rotation]
        _1d_right_environment = zeros((3,) + (bandwidth,)*2,dtype=complex128)
        for tag, value in self._2d.sides[self.rotation].items():
            _1d_right_environment[slot_of[tag]] = value.toArray().reshape(bandwidth,bandwidth)
        return NDArrayData(_1d_right_environment)
    # }}}
    def copy2Dto1D(self): # {{{
        self._1d.setCenterState(self._2d.state_center_data.join((0,1),(2,3),4))
        self._1d.left_environment = self.convert2DLeftEnvironment()
        self._1d.right_environment = self.convert2DRightEnvironment()
    # }}}
    def getCenterStateAsArray(self): # {{{
        self.check("while getting center state as array")
        return self._1d.getCenterStateAsArray()
    # }}}
    def increaseBandwidth(self,direction,by=None,to=None,do_as_much_as_possible=False): # {{{
        self.check("before increasing bandwidth")
        #self._1d.increaseBandwidth(direction,by,to,do_as_much_as_possible)
        self._2d.increaseBandwidth(direction,by,to,do_as_much_as_possible)
        self.copy2Dto1D()
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
