# Imports {{{
from collections import namedtuple
from functools import partial

from .utils import L,R,O, buildTensor
# }}}

# Base classes {{{
class Singleton: # {{{
    """A utility class that when created returns a singleton."""
    __slots__ = []
    def __new__(cls):
        try:
            return cls.instance
        except AttributeError:
            instance = object.__new__(cls)
            cls.instance = instance
            return instance
    def __hash__(self):
        return hash(type)
# }}}
# }}}

# Tags {{{
class Identity(Singleton): # {{{
    """\
Indicates that the tagged component contains the expectation with respect to
the identity.\
"""
    __slots__ = []
    def __repr__(self):
        return "Identity()"
# }}}
class Complete(Singleton): # {{{
    """\
Indicates that every term in the tagged component contains the complete
non-trivial part of the operator.\
"""
    __slots__ = []
    def __repr__(self):
        return "Complete()"
# }}}
class OneSiteOperator: # {{{
    """\
Indicates that the tagged component is the expectation value of a one-site
operator; if there is going to be more than one such one-site operator in a
:class:`System` than use the *id* field to distinguish them.\
"""
    __slots__ = ["id"]
    def __init__(self,id=None): # {{{
        self.id = id
    # }}}
    def __eq__(self,other): # {{{
        return (
            isinstance(other,OneSiteOperator) and
            other.id == self.id
        )
    # }}}
    def __hash__(self): # {{{
        return (
            hash(OneSiteOperator) ^
            hash(self.id)
        )
    # }}}
    def __repr__(self): # {{{
        return "OneSiteOperator({})".format(self.id)
    # }}}
# }}}
class TwoSiteOperator: # {{{
    """\
Indicates that the tagged component is the expectation value of part of a
two-site operator, the other part of which is in *direction*; in corner and
side environment tensors the *position* field indicates the distance from the
center --- i.e., each contraction in a given direction will bump the position
of the relevant tags up by one.\
"""
    __slots__ = ["id","direction","position"]
    def __init__(self,id,direction,position=None): # {{{
        self.id = id
        self.direction = direction
        self.position = position
    # }}}
    def __eq__(self,other): # {{{
        return (
            isinstance(other,TwoSiteOperator) and
            other.id == self.id and
            other.direction == self.direction and
            other.position == self.position
        )
    # }}}
    def __hash__(self): # {{{
        return (
            hash(TwoSiteOperator) ^
            hash(self.id) ^
            hash(self.direction) ^
            hash(self.position)
        )
    # }}}
    def __repr__(self): # {{{
        return "TwoSiteOperator({},{},{})".format(self.id,self.direction,self.position)
    # }}}
    def matches(left,right): # {{{
        """Returns :class:`Complete` if *left* and *right* together form an entire two-site operator term and None otherwise."""
        if(
            left.id == right.id and
            left.direction == RIGHT and
            right.direction == LEFT and
            left.position == right.position
        ):
            return Complete()
    # }}}
    def matchesCenter(self,side_direction,center): # {{{
        """Returns :class:`Complete` if *self* (a side) and *center* together form an entire two-site operator and None otherwise."""
        if(
            self.id == center.id and
            self.direction == CENTER and
            side_direction == center.direction
        ):
            return Complete()
    # }}}
    def matchesCenterIdentity(self): # {{{
        """Returns a copy of *self* with the position advanced by one if it is not 'broken' by putting an identity operator below it, which is true as long as the direction field is not CENTER, and None otherwise."""
        if self.direction != CENTER:
            return self.moveOut()
    # }}}
    def matchesCenterForStage3(self,direction,other): # {{{
        return (
            self.id == other.id and
            self.direction == CENTER and
            self.position + 2*direction == other.direction
        )
    # }}}
    def matchesCornerIdentityOnLeft(self): # {{{
        if self.direction == RIGHT:
            return self
        if self.direction == CENTER:
            return self.withNewDirectionAndPosition(LEFT,0)
    # }}}
    def matchesCornerIdentityOnLeftForStage1(self): # {{{
        if self.direction in (RIGHT,CENTER):
            return self
    # }}}
    def matchesCornerIdentityOnRight(self): # {{{
        if self.direction == LEFT:
            return self
        if self.direction == CENTER:
            return self.withNewDirectionAndPosition(RIGHT,0)
    # }}}
    def matchesSideIdentityOnLeft(self): # {{{
        if self.direction == RIGHT:
            return self.moveOut()
    # }}}
    def matchesSideIdentityOnRight(self): # {{{
        if self.direction == LEFT:
            return self.moveOut()
    # }}}
    def matchesSideIdentityOnRightForStage1(self): # {{{
        if self.direction == LEFT:
            return self
    # }}}
    def matchesSideIdentityOutward(self,direction): # {{{
        if self.direction == L(direction):
            return self.withNewDirectionAndPosition(LEFT,0)
        if self.direction == R(direction):
            return self.withNewDirectionAndPosition(RIGHT,0)
        if self.direction == O(direction):
            return self.withNewDirectionAndPosition(CENTER)
    # }}}
    def matchesStage1IdentityOnLeft(self): # {{{
        if self.direction == RIGHT:
            return self
        if self.direction == CENTER:
            return self.withNewDirectionAndPosition(CENTER,LEFT)
    # }}}
    def matchesStage1IdentityOnRight(self): # {{{
        if self.direction == LEFT:
            return self
        if self.direction == CENTER:
            return self.withNewDirectionAndPosition(CENTER,RIGHT)
    # }}}
    def matchesForStage3(x,y): # {{{
        return (
            x.id == y.id and
            (x.direction,y.direction) in ((LEFT,RIGHT),(RIGHT,LEFT)) and
            x.position == y.position
        )
    # }}}
    def moveOut(self): # {{{
        return self.withNewDirectionAndPosition(self.direction,self.position+1)
    # }}}
    def withNewDirectionAndPosition(self,direction,position=None): # {{{
        return TwoSiteOperator(self.id,direction,position)
    # }}}
# }}}
class TwoSiteOperatorCompressed: # {{{
    __slots__ = ["direction"]
    def __init__(self,direction): # {{{
        self.direction = direction
    # }}}
    def __eq__(self,other): # {{{
        return isinstance(other,TwoSiteOperatorCompressed) and other.direction == self.direction
    # }}}
    def __hash__(self): # {{{
        return hash(TwoSiteOperatorCompressed) ^ hash(self.direction)
    # }}}
    def __repr__(self): # {{{
        return "TwoSiteOperatorCompressed({})".format(self.direction)
    # }}}
    def matches(left,right): # {{{
        if left.direction == RIGHT and right.direction == LEFT:
            return Complete()
    # }}}
    def matchesCornerIdentityOnLeftForStage1(self): # {{{
        if self.direction == RIGHT:
            return self
    # }}}
    def matchesSideIdentityOnRightForStage1(self): # {{{
        if self.direction == LEFT:
            return self
    # }}}
    def matchesStage1IdentityOnLeft(self): # {{{
        if self.direction == RIGHT:
            return self
    # }}}
    def matchesStage1IdentityOnRight(self): # {{{
        if self.direction == LEFT:
            return self
    # }}}
    def matchesForStage3(x,y): # {{{
        return (x.direction,y.direction) in ((LEFT,RIGHT),(RIGHT,LEFT))
    # }}}
# }}}
# }}}

# Directions {{{
LEFT = 0
RIGHT = 1
CENTER = 2
#}}}

# Functions {{{
def addStandardCompleteAndIdentityTerms(terms,dense): # {{{
    terms[Identity,Identity] = lambda x,y: (Identity(),dense)
    terms[Complete,Identity] = lambda x,y: (Complete(),dense)
    terms[Identity,Complete] = lambda x,y: (Complete(),dense)
# }}}
def contractSparseTensors(dense_contractors,tensor_1,tensor_2): # {{{
    result_tensor = {}
    for tag_1, data_1 in tensor_1.items():
        tag_1_type = type(tag_1)
        for tag_2, data_2 in tensor_2.items():
            tag_2_type = type(tag_2)
            if (tag_1_type,tag_2_type) in dense_contractors:
                handler = dense_contractors[tag_1_type,tag_2_type](tag_1,tag_2)
                if handler is not None:
                    result_tag, contractDenseTensors = handler
                    if result_tag is not None:
                        result_data = contractDenseTensors(data_1,data_2)
                        if result_tag in result_tensor:
                            result_tensor[result_tag] += result_data
                        else:
                            result_tensor[result_tag]  = result_data
    return result_tensor
# }}}
def directSumListsOfSparse(list1,list2): # {{{
    return [directSumSparse(sparse1,sparse2) for (sparse1,sparse2) in zip(list1,list2)]
# }}}
def directSumSparse(sparse1,sparse2): # {{{
    sparses = (sparse1,sparse2)
    zeroses = tuple(sparse[Identity()].newZeros(sparse[Identity()].shape,dtype=sparse[Identity()].dtype) for sparse in sparses)
    result = {}
    for tag in sparse1.keys() | sparse2.keys():
        sparse_datas = tuple(
            sparse[tag] if tag in sparse else zeros
            for sparse, zeros in zip(sparses,zeroses)
        )
        result[tag] = sparse_datas[0].directSumWith(sparse_datas[1])
    return result
# }}}
def formSparseContractor(dense_contractors): # {{{
    return partial(contractSparseTensors,dense_contractors)
# }}}
def getInformationFromOperatorCenter(operator_center): # {{{
    for matrix in operator_center.values():
        return matrix.shape[0], matrix.dtype, type(matrix)
    else:
        raise ValueError("operator tensor has no term from which to extract the physical dimension")
# }}}
def makeMPO(I,Os=[],OOs=[]): # {{{
    size = 2 + len(OOs)
    last = size - 1
    components = {
        (0,0): I,
        (last,last): I,
    }
    for O in Os:
        if (0,last) in components:
            components[0,last] += O
        else:
            components[0,last] = O
    for id, OO in enumerate(OOs):
        components[0,id+1] = OO[1]
        components[id+1,last] = OO[0]
    return (
        buildTensor((size,size,len(I),len(I)),components),
        [1] + [0]*(size-1),
        [Identity()] + [TwoSiteOperator(id,2) for id in range(len(OOs))] + [Complete()],
        [0]*(size-1) + [1],
        [Complete()] + [TwoSiteOperator(id,2) for id in range(len(OOs))] + [Identity()],
    )
# }}}
def makeSimpleSparseOperator(O=None,OO_UD=None,OO_LR=None): # {{{
    return makeSparseOperator(
        [O] if O is not None else [],
        [OO_UD] if OO_UD is not None else [],
        [OO_LR] if OO_LR is not None else [],
    )
# }}}
def makeSparseOperator(Os=[],OO_UDs=[],OO_LRs=[]): # {{{
    operator = {}
    identity = None

    def checkAgainstIdentityShape(shape):
        nonlocal identity_shape
        if identity_shape != shape:
            raise ValueError("incompatible site matrix shapes: {} and {}".format(identity_shape,shape))

    for id, O in enumerate(Os): # {{{
        operator[OneSiteOperator(id)] = O
        if identity is None:
            identity = O.newIdentity(O.shape[0])
            identity_shape = O.shape
        else:
            checkAgainstIdentityShape(O.shape)
    # }}}

    for id, (O_U, O_D) in enumerate(OO_UDs): # {{{
        operator[TwoSiteOperator(id,3,0)] = O_U
        operator[TwoSiteOperator(id,1,0)] = O_D
        if identity is None:
            identity = O_U.newIdentity(O_U.shape[0])
            identity_shape = O_U.shape
        else:
            checkAgainstIdentityShape(O_U.shape)
        checkAgainstIdentityShape(O_D.shape)
    # }}}

    for id, (O_L, O_R) in enumerate(OO_LRs): # {{{
        operator[TwoSiteOperator(id,0,0)] = O_L
        operator[TwoSiteOperator(id,2,0)] = O_R
        if identity is None:
            identity = O_L.newIdentity(O_L.shape[0])
            identity_shape = O_L.shape
        else:
            checkAgainstIdentityShape(O_L.shape)
        checkAgainstIdentityShape(O_R.shape)
    # }}}

    if identity is None:
        raise ValueError("No terms have been specified.")
    operator[Identity()] = identity
    return operator
# }}}
def mapOverSparseData(f,sparse): # {{{
    return {tag: f(data) for tag, data in sparse.items()}
# }}}
def stripAllButIdentityFrom(sparse): # {{{
    return {Identity():sparse[Identity()]}
# }}}
# }}}

# Exports {{{
__all__ = [
    "Identity",
    "Complete",
    "OneSiteOperator",
    "TwoSiteOperator",
    "TwoSiteOperatorCompressed",

    "addStandardCompleteAndIdentityTerms",
    "contractSparseTensors",
    "directSumListsOfSparse",
    "directSumSparse",
    "formSparseContractor",
    "getInformationFromOperatorCenter",
    "makeMPO",
    "makeSimpleSparseOperator",
    "mapOverSparseData",
    "stripAllButIdentityFrom",
]
# }}}
