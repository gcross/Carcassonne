# Imports {{{
from functools import partial
from numpy import prod

from ..data import NDArrayData
from ..data.cost_tracker import computeCostOfContracting
from ..utils import Join, Multiplier, formContractor, prependDataContractor
# }}}

# Functions {{{
# def absorbCenterOSSIntoLeftEnvironment(L,O,S,S*) {{{
absorbCenterOSSIntoLeftEnvironment = formContractor(
    ["L","O","S","S*"],
    [
        (("L",0),("O",0)),
        (("L",1),("S",0)),
        (("L",2),("S*",0)),
        (("O",2),("S*",2)),
        (("O",3),("S",2)),
    ],[
        [("O",1)],
        [("S",1)],
        [("S*",1)],
    ]
) # }}}
# def absorbCenterOSSIntoRightEnvironment(R,O,S,S*) {{{
absorbCenterOSSIntoRightEnvironment = formContractor(
    ["R","O","S","S*"],
    [
        (("R",0),("O",1)),
        (("R",1),("S",1)),
        (("R",2),("S*",1)),
        (("O",2),("S*",2)),
        (("O",3),("S",2)),
    ],[
        [("O",0)],
        [("S",0)],
        [("S*",0)],
    ]
) # }}}
# def absorbCenterSSIntoRightEnvironment(R,O,S,S*) {{{
absorbCenterSSIntoRightEnvironment = formContractor(
    ["R","S","S*"],
    [
        (("R",0),("S",1)),
        (("R",1),("S*",1)),
        (("S",2),("S*",2)),
    ],[
        [("S",0)],
        [("S*",0)],
    ]
) # }}}
# def formExpectationMultiplier(L,R,O,S) {{{
@prependDataContractor(
    [
        Join(0,0,2,0),
        Join(1,0,2,1),
    ],[
        [(0,2),(1,2),(2,2)],
        [(0,1),(1,1),(2,3)],
    ]
)
@prependDataContractor(
    [
        Join(0,0,2,0),
        Join(0,1,3,0),
        Join(1,0,2,1),
        Join(1,1,3,1),
        Join(2,3,3,2),
    ],[
        [(0,2)],
        [(1,2)],
        [(2,2)],
    ]
)
def formExpectationMultiplier(multiply,formMatrix,L,R,O):
    L = NDArrayData(L)
    R = NDArrayData(R)
    O = NDArrayData(O)
    S_shape = (L.shape[1],R.shape[1],O.shape[2])
    return \
        Multiplier(
            (prod(S_shape),)*2,
            partial(multiply,L,R,O),
            computeCostOfContracting(multiply,L,R,O,S_shape),
            partial(formMatrix,L,R,O),
            computeCostOfContracting(formMatrix,L,R,O),
        )
# }}}
# }}}
