# Imports {{{
from functools import partial
from numpy import prod

from ..data import NDArrayData
from ..data.cost_tracker import computeCostOfContracting
from ..utils import Join, Multiplier, formDataContractor, prependDataContractor
# }}}

# Functions {{{
# def absorbCenterOSSIntoLeftEnvironment(L,O,S,S*) {{{
absorbCenterOSSIntoLeftEnvironment = formDataContractor(
    [
        Join(0,0,1,0),
        Join(0,1,2,0),
        Join(0,2,3,0),
        Join(1,2,3,2),
        Join(1,3,2,2),
    ],[
        [(1,1)],
        [(2,1)],
        [(3,1)],
    ]
) # }}}
# def absorbCenterOSSIntoRightEnvironment(R,O,S,S*) {{{
absorbCenterOSSIntoRightEnvironment = formDataContractor(
    [
        Join(0,0,1,1),
        Join(0,1,2,1),
        Join(0,2,3,1),
        Join(1,2,3,2),
        Join(1,3,2,2),
    ],[
        [(1,0)],
        [(2,0)],
        [(3,0)],
    ]
) # }}}
# def absorbCenterSSIntoRightEnvironment(R,S,S*) {{{
absorbCenterSSIntoRightEnvironment = formDataContractor(
    [
        Join(0,0,1,1),
        Join(0,1,2,1),
        Join(1,2,2,2),
    ],[
        [(1,0)],
        [(2,0)],
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
