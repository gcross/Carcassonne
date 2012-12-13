# Functions {{{
# def absorbCenterOSSIntoLeftEnvironment(L,O,S,S*) {{{
absorbCenterOSSIntoLeftEnvironment = formDataContractor(
    [
        Join(0,0,1,0),
        Join(0,1,2,0),
        Join(0,2,3,0),
        Join(1,3,2,2),
        Join(1,2,3,2),
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
        Join(1,3,2,2),
        Join(1,2,3,2),
    ],[
        [(1,2)],
        [(2,2)],
        [(3,2)],
    ]
) # }}}
# def formExpectationMultiplier(L,R,O) {{{
@prependDataContractor(
    [
        Join(0,0,2,0),
        Join(0,1,3,0),
        Join(1,0,2,1),
        Join(1,1,3,1),
        Join(3,2,2,3),
    ],[
        [(0,2)],
        [(1,2)],
        [(2,2)],
    ]
def formExpectationMultiplier(L,R,O):
# }}}
# }}}
