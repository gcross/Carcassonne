# Imports {{{
from collections import defaultdict
from functools import partial, reduce
from numpy import argmax, argmin, array, complex128, dot, identity, multiply, prod, sqrt, set_printoptions, tensordot, trace, zeros
from numpy.random import rand, random_sample
from scipy.linalg import LinAlgError, eig, eigh, eigvals, lu_factor, lu_solve, norm, svd, qr
from scipy.sparse.linalg import LinearOperator, eigs, eigsh, gmres
# }}}

set_printoptions(linewidth=132)

# Exceptions {{{
class DimensionMismatchError(ValueError): # {{{
    """\
Thrown when two tensors connected by a join have incompatable dimensions for the
indices of the join.

    * left_tensor_number
       the number of the left tensor in the join

    * left_index
        the index of the left tensor to be joined

    * left_dimension
        the dimension of the index *left_index* in the left tensor

    * right_tensor_number
       the number of the right tensor in the join

    * right_index
        the index of the right tensor to be joined

    * right_dimension
        the dimension of the index *right_index* in the right tensor\
"""
    def __init__(self,left_tensor_number,left_index,left_dimension,right_tensor_number,right_index,right_dimension): # {{{
        self.left_tensor_number = left_tensor_number
        self.left_index = left_index
        self.left_dimension = left_dimension
        self.right_tensor_number = right_tensor_number
        self.right_index = right_index
        self.right_dimension = right_dimension
        ValueError.__init__(self,"tensor {left_tensor_number}'s index {left_index} has dimension {left_dimension}, whereas tensor {right_tensor_number}'s index {right_index} has dimension {right_dimension}".format(**self.__dict__))
    # }}}
# }}}
class InvariantViolatedError(Exception): # {{{
    """Thrown whan an invariant has been violated."""
# }}}
class RelaxFailed(Exception): # {{{
    """\
Thrown when the eigensolver encountered a problem improving the initial
value; *initial_value* is the initial value fed into the eigenvalue solver and
*final_value* is the output of the solver, which may not be valid.\
"""
    def __init__(self,initial_value,final_value):
        self.initial_value = initial_value
        self.final_value = final_value
    def __str__(self):
        return "{} --> {}".format(self.initial_value,self.final_value)
    def __repr__(self):
        return "RelaxFailed({},{})".format(self.initial_value,self.final_value)
# }}}
class UnexpectedTensorRankError(ValueError): # {{{
    """\
Thrown when a tensor had an unexpected rank.

    * tensor_number
        the number of the tensor

    * expected_rank
        the expected rank of the tensor

    * actual_rank
        the actual rank of the tensor\
"""
    def __init__(self,tensor_number,expected_rank,actual_rank): # {{{
        self.tensor_number = tensor_number
        self.expected_rank = expected_rank
        self.actual_rank = actual_rank
        ValueError.__init__(self,"tensor {tensor_number} was expected to have rank {expected_rank} but actually has rank {actual_rank}".format(**self.__dict__))
    # }}}
# }}}
# }}}

# Classes {{{
class Join: # {{{
    """\
Represents a join between two tensors.

    * left_tensor_number
        the number of the left tensor in the join

    * left_tensor_indices
        the indices of the left tensor that are joined to the corresponding
        indices in the right tensor given by *left_tensor_indices*; if this is
        given as an integer then it is wrapped into a singleton list

    * right_tensor_number
        the number of the right tensor in the join

    * right_tensor_indices
        the indices of the right tensor that are joined to the corresponding
        indices in the right tensor given by *right_tensor_indices*; if this is
        given as an integer then it is wrapped into a singleton list

The constructor checks that the :class:`Join` is valid, i.e. that it is not a
self-join (where the left and right tensors are the same) and that the number
of indices of the left tensor matches that of the right tensor.\
"""
    def __init__( # {{{
        self
        ,left_tensor_number
        ,left_tensor_indices
        ,right_tensor_number
        ,right_tensor_indices
    ):
        if not hasattr(left_tensor_indices,"__len__") or isinstance(left_tensor_indices,str):
            left_tensor_indices = [left_tensor_indices]
        if not hasattr(right_tensor_indices,"__len__") or isinstance(right_tensor_indices,str):
            right_tensor_indices = [right_tensor_indices]
        if len(left_tensor_indices) != len(right_tensor_indices):
            raise ValueError("number of left indices does not match number of right indices (len({}) != len({}))".format(left_tensor_indices,right_tensor_indices))
        self.left_tensor_number = left_tensor_number
        self.left_tensor_indices = left_tensor_indices
        self.right_tensor_number = right_tensor_number
        self.right_tensor_indices = right_tensor_indices
        self.checkOrderAndSwap()
    # }}}
    def __repr__(self): # {{{
        return "Join({left_tensor_number},{left_tensor_indices},{right_tensor_number},{right_tensor_indices})".format(**self.__dict__)
    # }}}
    def checkOrderAndSwap(self): # {{{
        """\
Asserts that the :class:`Join` is not a self-join and then ensures that the
*left_tensor_number* is less than the *right_tensor_number* by swapping if
necessary.\
"""
        if self.left_tensor_number == self.right_tensor_number:
            raise ValueError(
                ("attempted to create a self-join for tensor {} (indices = {},{})"
                ).format(
                    self.left_tensor_number,
                    self.left_tensor_indices,
                    self.right_tensor_indices
                )
            )
        if self.right_tensor_number < self.left_tensor_number:
            right_tensor_number = self.right_tensor_number
            right_tensor_indices = self.right_tensor_indices
            self.right_tensor_number = self.left_tensor_number
            self.right_tensor_indices = self.left_tensor_indices
            self.left_tensor_number = right_tensor_number
            self.left_tensor_indices = right_tensor_indices
    # }}}
    def mergeWith(self,other): # {{{
        """\
Merges the join indices of two joins that involve the same two tensors;  if the
tensor numbers in the two joins do not match then an error is raised.\
"""
        assert self.left_tensor_number == other.left_tensor_number
        assert self.right_tensor_number == other.right_tensor_number
        self.left_tensor_indices  += other.left_tensor_indices
        self.right_tensor_indices += other.right_tensor_indices
    # }}}
    def update(self,old_tensor_number,new_tensor_number,index_map): # {{{
        """\
Updates the tensor number of the matching tensor from *old_tensor_number* to
*new_tensot_number*, and also updates its join indices using the given
*index_map*;  if neither the left nor the right tensor numbers match
*old_tensor_number* than an error is raised.\
"""
        if self.left_tensor_number == old_tensor_number:
            self.left_tensor_number = new_tensor_number
            self.left_tensor_indices = applyIndexMapTo(index_map,self.left_tensor_indices)
            self.checkOrderAndSwap()
        elif self.right_tensor_number == old_tensor_number:
            self.right_tensor_number = new_tensor_number
            self.right_tensor_indices = applyIndexMapTo(index_map,self.right_tensor_indices)
            self.checkOrderAndSwap()
    # }}}
# }}}
class Multiplier: # {{{
    """\
Represents a matrix-vector multiplication operation.

    * shape
        the shape of the matrix

    * multiply
        a function that applies this multipliction operation to a given vector

    * cost_of_multiply
        the cost of calling *multiply*

    * formMatrix
        a function that computes the explicit matrix representation of this
        multiplication operation

    * cost_of_formMatrix
        the cost of calling *formMatrix*

The unit of the two costs does not matter as long as they are consistent;  they
are used to estimate whether using the given *multiply* function is cheaper
(for a given number of applications) than explicitly constructing and using the
matrix representation of this operation.\
"""
    def __init__(self,shape,multiply,cost_of_multiply,formMatrix,cost_of_formMatrix): # {{{
        self.shape = shape
        self.multiply = multiply
        self.cost_of_multiply = cost_of_multiply
        self.formMatrix = formMatrix
        self.cost_of_formMatrix = cost_of_formMatrix
    # }}}
    def __call__(self,vector): # {{{
        """Applies this object's *multiply* function to *vector*."""
        return self.multiply(vector)
    # }}}
    @classmethod # fromMatrix {{{
    def fromMatrix(self,matrix):
        """\
Constructs a :class:`Multiplier` whose multiplication operator is given by the
explicit *matrix*.\
"""
        m, n = matrix.shape
        return \
            Multiplier(
                matrix.shape,
                lambda v: matrix.matvecWith(v),
                m*n,
                lambda: matrix,
                0
            )
    # }}}
    def isCheaperToFormMatrix(self,estimated_number_of_applications): # {{{
        """\
Computes whether it would be cheaper to form the explicit matrix representation
of this multiplication if it is going to be used at least
*estimated_number_of_applications*.\
"""
        return estimated_number_of_applications*self.cost_of_multiply > \
                self.cost_of_formMatrix + estimated_number_of_applications*self.shape[0]*self.shape[1]
    # }}}
# }}}
# }}}

# Decorators {{{
class prepend: # {{{
    """\
This decorator takes the arguments and keywords given to it and passes them to
the decorated function as its first arguments and keywords.\
"""
    def __init__(self,*args,**keywords):
        self.args = args
        self.keywords = keywords
    def __call__(self,f):
        return partial(f,*self.args,**self.keywords)
# }}}
def prependDataContractor(*args,**keywords): # {{{
    """\
This decorator chains :func:`prepend` with the result of calling
:func:`formDataContractor` with the given arguments and keywords.\
"""
    return prepend(formDataContractor(*args,**keywords))
# }}}
# }}}

# Pauli Operators {{{
class Pauli:
    """\
This class is a namespace which contains the four (unnormalized) Pauli
operators: *I*, *X*, *Y*, and *Z*.\
"""
    I = identity(2,dtype=complex128)
    X = array([[0,1],[1,0]],dtype=complex128)
    Y = array([[0,-1j],[1j,0]],dtype=complex128)
    Z = array([[1,0],[0,-1]],dtype=complex128)
# }}}

# Functions {{{
def applyIndexMapTo(index_map,indices): # {{{
    """\
Applies the given *index_map* to the given *indices* --- that is, it constructs
a new list by evaluating the map *index_map* at each index in *indices*.\
"""
    return [index_map[index] for index in indices]
# }}}
def applyPermutation(permutation,values): # {{{
    """\
Applies the given *permutation* to the given *values* --- that is, it
constructs a new list by looking up ``values[i]`` for each ``i`` in
*permutation*.\
"""
    return [values[i] for i in permutation]
# }}}
def buildProductTensor(*factors): # {{{
    """Builds a tensor by taking the outer product of all the *factors*."""
    return reduce(multiply.outer,(array(factor,dtype=complex128) for factor in factors)) #,zeros((),dtype=complex128))
# }}}
def buildTensor(dimensions,components): # {{{
    """\
Buikds a sparse tensor by creating a zero tensor with the given dimensions and
then filling it with the values in *components*, which keys must be tuples that
are valid indices of the tensor.\
"""
    tensor = zeros(dimensions,dtype=complex128)
    for index, value in components.items():
        tensor[index] = value
    return tensor
# }}}
def checkForNaNsIn(data): # {{{
    """Asserts that *data* has no `NaN` and then returns it."""
    assert not data.hasNaN()
    return data
# }}}
def crand(*shape): # {{{
    """Returns a tensor with the given shape filled with random complex numers."""
    return rand(*shape)*2-1+rand(*shape)*2j-1j
# }}}
def computeAndCheckNewDimension(shape,direction,by=None,to=None,do_as_much_as_possible=False): # {{{
    """\
Computes a new shape from *shape* by increasing its value at the index
*direction* as much as possible either by the value given in *by* or to the
value given in *to*.  If the new value of this dimension is large enough that
it is no longer possible to normalize the tensor along *direction* (which
happens when the number of degrees of freedom in *direction* is larger than the
product of the dimensions in all other directions), then if
*do_as_much_as_possible* is set to True it will set it to the largest possible
dimension that allows for normalization and otherwise it will raise a
ValueError.\
"""
    old_dimension = shape[direction]
    new_dimension = computeNewDimension(old_dimension,by=by,to=to)
    maximum_new_dimension = maximumNewBandwidth(direction,shape)
    if new_dimension > maximum_new_dimension:
        if do_as_much_as_possible:
            new_dimension = maximum_new_dimension
        else:
            raise ValueError("New dimension {} is greater than the current maximum dimension for direction {}, which is {} (shape = {}).".format(new_dimension,direction,shape,maximum_new_dimension,shape))
    return old_dimension, new_dimension, new_dimension-old_dimension
# }}}
def computeCompressor(new_dimension,multiplier,dtype): # {{{
    """\
Computes the matrix that optimally compresses the Hermetian matrix whose
matrix-vector multiplication operation is given by *multiplier*, which must be
of type :class:`Multiplier`, from its current dimension to *new_dimension*.
That is, if X is a matrix with dimensions (o,o) whose multiplication operation
is given by *multiplier*, then this function returns a matrix C with dimensions
(*new_dimension*,o) such that C * C^H is the identity, and C * X * C^H is the
best (*new_dimension*,*new_dimension*) approximation to X.\
"""
    old_dimension = multiplier.shape[0]
    assert old_dimension == multiplier.shape[1]
    if new_dimension < 0:
        raise ValueError("New dimension ({}) must be non-negative.".format(new_dimension))
    elif new_dimension > old_dimension:
        raise ValueError("New dimension ({}) must be less than or equal to the old dimension ({}).".format(new_dimension,old_dimension))
    elif new_dimension == 0:
        return (zeros((new_dimension,old_dimension),dtype=dtype),)*2
    elif new_dimension >= old_dimension // 2:
        matrix = multiplier.formMatrix()
        if tuple(matrix.shape) != (old_dimension,)*2:
            raise ValueError("Multiplier matrix has shape {} but the old dimension is {}.".format(matrix.shape,old_dimension))
        evals, evecs = eigh(matrix)
        evals = evals[-new_dimension:]
        evecs = evecs[:,-new_dimension:]
    else:
        operator = \
            LinearOperator(
                shape=(old_dimension,)*2,
                matvec=multiplier,
                dtype=dtype
            )
        evals, evecs = eigsh(operator,k=new_dimension)
    evecs = evecs.transpose()
    while new_dimension > 0 and abs(evals[new_dimension-1]) < 1e-15:
        new_dimension -= 1
    if new_dimension == 0:
        raise ValueError("Input is filled with near-zero elements.")
    return evecs
# }}}
def computeCompressorForMatrixTimesItsDagger(new_dimension,matrix): # {{{
    """This is a convenience function that, given a matrix X, calls :func:`computeCompressor` with the matrix X^H * X."""
    other_dimension = matrix.shape[0]
    matrix_dagger = matrix.transpose().conj()
    return \
        computeCompressor(
            new_dimension,
            Multiplier(
                (old_dimension,)*2,
                lambda v: dot(matrix_dagger,dot(matrix,v)),
                2 * old_dimension * other_dimension,
                lambda: dot(matrix_dagger,matrix),
                old_dimension**2 * other_dimension
            ),
            matrix.dtype,
        )
# }}}
def computeLengthAndCheckForGaps(indices,error_message): # {{{
    """\
Checks that *indices* contains every number between 0 and ``len(indices)-1``
exactly once --- i.e., that it represents a permutation;  if the check fails
then a :exc:`ValueError` is raised which includes the message provided in
*error_message*.\
"""
    if len(indices) == 0:
        return 0
    length = max(indices)+1
    unobserved_indices = set(range(length))
    for index in indices:
        unobserved_indices.remove(index)
    if unobserved_indices:
        raise ValueError(error_message + ": " + str(unobserved_indices))
    return length
# }}}
def computeAbsoluteLimitingLinearCoefficient(n,multiplyO,multiplyN,multiplyL,multiplyR): # {{{
    """\
Given an infinite MPO and MPS, returns the rate at which the expectation value grows as new sites are added to the system.

multiplyO
    the action of the MPO on the right operator left environment

multiplyN
    the action of the normalization MPO on the right normalization left environment

multiplyL
    the result of applying the given matrix to left environment

multiplyR
    the result of applying the given matrix to right environment\
"""
    if True: # n <= 3:
        matrix = []
        for i in range(n):
            matrix.append(multiplyO(array([0]*i+[1]+[0]*(n-1-i))))
        matrix = array(matrix)
        evals = eigvals(matrix)
        lam = evals[argmax(abs(evals))]
        tmatrix = matrix-lam*identity(n)
        ovecs = svd(dot(tmatrix,tmatrix))[-1][-2:]
        assert ovecs.shape == (2,n)
    else:
        ovals, ovecs = eigs(LinearOperator((n,n),matvec=multiplyO),k=2,which='LM',ncv=9)
        ovecs = ovecs.transpose()

    Omatrix = zeros((2,2),dtype=complex128)
    for i in range(2):
        for j in range(2):
            Omatrix[i,j] = dot(ovecs[i].conj(),multiplyO(ovecs[j]))
    numerator = sqrt(trace(dot(Omatrix.transpose().conj(),Omatrix))-2)

    lnvecs = multiplyL(ovecs)
    rnvecs = multiplyR(ovecs)
    Nmatrix = zeros((2,2),dtype=complex128)
    for i in range(2):
        for j in range(2):
            Nmatrix[i,j] = dot(lnvecs[i].conj(),multiplyN(rnvecs[j]))
    denominator = sqrt(trace(dot(Nmatrix.transpose().conj(),Nmatrix)))
    return numerator/denominator
# }}}
def computeNewDimension(old_dimension,by=None,to=None): # {{{
    """\
Computes the result of either adding *by* to *old_dimension* or returning *to*,
i.e. so that this function either increase *old_dimension* by *by* or increases
it to *to*.  Errors will be raised if either both *by* and *to* or neither has
been given, and if the new dimension is smaller than *old_dimension*.

This functions is not very useful as it own, but it exists to make it easier to
write other functions taking *by* and *to* arguments.\
"""
    if by is None and to is None:
        raise ValueError("Either 'by' or 'to' must not be None.")
    elif by is not None and to is not None:
        raise ValueError("Both 'by' ({}) and 'to' ({}) cannot be None.".format(by,to))
    elif by is not None:
        new_dimension = old_dimension + by
    elif to is not None:
        new_dimension = to
    assert new_dimension >= old_dimension
    return new_dimension
# }}}
def computePostContractionIndexMap(rank,contracted_indices,offset=0): # {{{
    contracted_indices = set(contracted_indices)
    index_map = dict()
    new_index = 0
    for old_index in range(rank):
        if old_index not in contracted_indices:
            index_map[old_index] = new_index + offset
            new_index += 1
    return index_map
# }}}
def dropAt(iterable,index): # {{{
    """\
Drops the value at *index* in *iterable*; if possible it constructs a value
with the same type as *iterable* and if not it returns a tuple.\
"""
    new_values = (x for i, x in enumerate(iterable) if i != index)
    try:
        return type(iterable)(new_values)
    except TypeError:
        return tuple(new_values)
# }}}
def formDataContractor(joins,final_groups,tensor_ranks=None): # {{{
    """\
Returns a function that takes one or more tensors as arguments and returns the
result of contracting the tensor network specified by *joins* and
*final_groups*.  For example, the following code returns a function that takes
two matrices and multiplies them together using standard matrix multiplication:

    ``formDataContractor([Join(0,1,1,0)],[[(0,0)],[(1,1)]])``

The parameters of this function are

joins
    a sequence of :class:`Join`s which specifies how the tensors are connected
    to each other

final_groups
    a sequence of sequences of tensor number/index tuples that specifies the
    final result --- that is, each element of *final_groups* corresponds to an
    axis of the final tensor and specifies which input tensor axes are merged
    together to form that axis

tensor_ranks
    if not `None`, provides a sequence that gives the rank of each tensor,
    which is used to check that the tensor ranks inferred from the other
    arguments are correct\
"""
    # Tabulate all of the tensor indices to compute the number of arguments and their ranks {{{
    observed_tensor_indices = defaultdict(set)
    observed_joins = set()
    def observeTensorIndices(tensor_number,*indices):
        if tensor_number < 0:
            raise ValueError("tensor numbers must be non-negative, not {}".format(tensor_number))
        observed_indices = observed_tensor_indices[tensor_number]
        for index in indices:
            if index in observed_indices:
                raise ValueError("index {} of tensor {} appears twice".format(index,tensor_number))
            observed_indices.add(index)
    for join in joins:
        observeTensorIndices(join.left_tensor_number,*join.left_tensor_indices)
        observeTensorIndices(join.right_tensor_number,*join.right_tensor_indices)
        if (join.left_tensor_number,join.right_tensor_number) in observed_joins:
            raise ValueError("two joins appear between tensors {} and {} (recall that specified joins can contain multiple indices)".format(join.left_tensor_number,join.right_tensor_number))
        else:
            observed_joins.add((join.left_tensor_number,join.right_tensor_number))
    for group in final_groups:
        for (tensor_number,index) in group:
            observeTensorIndices(tensor_number,index)
    number_of_tensors = computeLengthAndCheckForGaps(
        observed_tensor_indices.keys(),
        "the following tensor numbers were expected but not observed"
    )
    if number_of_tensors == 0:
        raise ValueError("no tensors have been specified to be contracted")
    observed_tensor_ranks = [
        computeLengthAndCheckForGaps(
            observed_tensor_indices[tensor_number],
            "the following indices of tensor {} were expected but not observed".format(tensor_number)
        )
        for tensor_number in range(number_of_tensors)
    ]
    if tensor_ranks is None:
        tensor_ranks = observed_tensor_ranks
    else:
        if tensor_ranks != observed_tensor_ranks:
            raise ValueError("the ranks of the arguments were specified to be {}, but inferred to be {}".format(tensor_ranks,observed_tensor_ranks))
    # }}}
    # Build the prelude for the function {{{
    function_lines = [
        "def contract(" + ",".join(["_{}".format(tensor_number) for tensor_number in range(number_of_tensors)]) + "):",
    ]
    # Build the documentation string {{{
    function_lines.append('"""')
    for tensor_number in range(number_of_tensors):
        function_lines.append("_{} - tensor of rank {}".format(tensor_number,tensor_ranks[tensor_number]))
    function_lines.append('"""')
    # }}}
    # Check that the tensors have the correct ranks {{{
    for tensor_number, expected_rank in enumerate(tensor_ranks):
        function_lines.append("if _{tensor_number}.ndim != {expected_rank}: raise UnexpectedTensorRankError({tensor_number},{expected_rank},_{tensor_number}.ndim)".format(tensor_number=tensor_number,expected_rank=expected_rank))
    # }}}
    # Check that the tensors have matching dimensions {{{
    for join in joins:
        for (left_index,right_index) in zip(join.left_tensor_indices,join.right_tensor_indices):
            function_lines.append('if _{left_tensor_number}.shape[{left_index}] != _{right_tensor_number}.shape[{right_index}]: raise DimensionMismatchError({left_tensor_number},{left_index},_{left_tensor_number}.shape[{left_index}],{right_tensor_number},{right_index},_{right_tensor_number}.shape[{right_index}])'.format(
                left_tensor_number = join.left_tensor_number,
                left_index = left_index,
                right_tensor_number = join.right_tensor_number,
                right_index = right_index,
            ))
    # }}}
    # }}}
    # Build the main part of the function {{{
    next_tensor_number = number_of_tensors
    active_tensor_numbers = set(range(number_of_tensors))
    joins.reverse()
    while joins:
        join = joins.pop()
        left_tensor_number = join.left_tensor_number
        left_tensor_indices = join.left_tensor_indices
        right_tensor_number = join.right_tensor_number
        right_tensor_indices = join.right_tensor_indices
        active_tensor_numbers.remove(left_tensor_number)
        active_tensor_numbers.remove(right_tensor_number)
        active_tensor_numbers.add(next_tensor_number)
        tensor_ranks.append(tensor_ranks[left_tensor_number]+tensor_ranks[right_tensor_number]-2*len(left_tensor_indices))
        # Add the lines to the function {{{
        function_lines.append("_{} = _{}.contractWith(_{},{},{})".format(
            next_tensor_number,
            left_tensor_number,
            right_tensor_number,
            left_tensor_indices,
            right_tensor_indices,
        ))
        function_lines.append("del _{}, _{}".format(left_tensor_number,right_tensor_number))
        # }}}
        # Update the remaining joins {{{
        left_index_map = computePostContractionIndexMap(tensor_ranks[left_tensor_number],left_tensor_indices)
        right_index_map = computePostContractionIndexMap(tensor_ranks[right_tensor_number],right_tensor_indices,len(left_index_map))
        i = len(joins)-1
        observed_joins = dict()
        while i >= 0:
            join_to_update = joins[i]
            join_to_update.update(left_tensor_number,next_tensor_number,left_index_map)
            join_to_update.update(right_tensor_number,next_tensor_number,right_index_map)
            observed_join_tensor_numbers = (join_to_update.left_tensor_number,join_to_update.right_tensor_number)
            if observed_join_tensor_numbers in observed_joins:
                observed_joins[observed_join_tensor_numbers].mergeWith(join_to_update)
                del joins[i]
            else:
                observed_joins[observed_join_tensor_numbers] = join_to_update
            i -= 1
        # }}}
        # Update the final groups {{{
        for group in final_groups:
            for i, (tensor_number,tensor_index) in enumerate(group):
                if tensor_number == left_tensor_number:
                    group[i] = (next_tensor_number,left_index_map[tensor_index])
                elif tensor_number == right_tensor_number:
                    group[i] = (next_tensor_number,right_index_map[tensor_index])
        # }}}
        next_tensor_number += 1
    # }}}
    # Build the finale of the function {{{
    # Combine any remaining tensors using outer products {{{
    active_tensor_numbers = list(active_tensor_numbers)
    final_tensor_number = active_tensor_numbers[0]
    for tensor_number in active_tensor_numbers[1:]: 
        function_lines.append("_{final_tensor_number} = _{final_tensor_number}.contractWith(_{tensor_number},[],[])".format(final_tensor_number=final_tensor_number,tensor_number=tensor_number))
        function_lines.append("del _{}".format(tensor_number))
    # }}}
    if len(final_groups) > 0:
        # Compute index map and apply the index map to the final groups {{{
        index_offset = 0
        index_map = {}
        for tensor_number in active_tensor_numbers:
            for index in range(tensor_ranks[tensor_number]):
                index_map[(tensor_number,index)] = index+index_offset
            index_offset += tensor_ranks[tensor_number]
        final_groups = [applyIndexMapTo(index_map,group) for group in final_groups]
        # }}}
        function_lines.append("return _{}.join(*{})".format(final_tensor_number,final_groups))
    else:
        function_lines.append("return _{}.extractScalar()".format(final_tensor_number))
    # }}}
    # Compile and return the function {{{
    function_source = "\n    ".join(function_lines)
    captured_definition = {}
    visible_exceptions = {}
    for exception_name in ["DimensionMismatchError","UnexpectedTensorRankError"]:
        visible_exceptions[exception_name] = globals()[exception_name]
    exec(function_source,visible_exceptions,captured_definition)
    contract = captured_definition["contract"]
    contract.source = function_source
    return contract
    # }}}
# }}}
def invertPermutation(permutation): # {{{
    """Inverts the given *permutation*."""
    return [permutation.index(i) for i in range(len(permutation))]
# }}}
def maximumNewBandwidth(direction,shape): # {{{
    """\
Returns the maximum possible bandwidth along *direction* in *shape* such that
the tensor is still normalizable in that direction.\
"""
    return prod(dropAt(shape,direction))
# }}}
def multiplyBySingleSiteOperator(state,operator): # {{{
    """\
Given a 2D PEPS in *state*, returns the result of applying the single-site
*operator* to *state* (along the physical axis).\
"""
    return state.absorbMatrixAt(4,operator)
# }}}
def multiplyTensorByMatrixAtIndex(tensor,matrix,index): # {{{
    """\
Multiplies the *tensor* at *index* by *matrix*, i.e. it contracts *tensor*'s
axis *index* with *matrix*'s first axis.\
"""
    tensor_new_indices = list(range(tensor.ndim-1))
    tensor_new_indices.insert(index,tensor.ndim-1)
    return tensordot(tensor,matrix,(index,0)).transpose(tensor_new_indices)
# }}}
def normalize(tensor,index): # {{{
    """\
Returns the normalization of *tensor* at *index* --- that is, the closest
tensor to *tensor* such that contracting it with its conjugate along all axes
except *index* results in the identity matrix.\
"""
    new_indices = list(range(tensor.ndim))
    del new_indices[index]
    new_indices.append(index)

    old_shape = list(tensor.shape)
    del old_shape[index]
    new_shape = (prod(old_shape),tensor.shape[index])
    old_shape.append(tensor.shape[index])

    new_tensor = tensor.transpose(new_indices).reshape(new_shape)
    if new_tensor.shape[1] > new_tensor.shape[0]:
        raise ValueError("There are not enough degrees of freedom available to normalize the tensor.")

    old_indices = list(range(tensor.ndim-1))
    old_indices.insert(index,tensor.ndim-1)

    try:
        u, s, v = svd(new_tensor,full_matrices=0)
        return dot(u,v).reshape(old_shape).transpose(old_indices)
    except LinAlgError:
        M = dot(new_tensor.conj().transpose(),new_tensor)

        vals, U = eigh(M)
        vals[vals<0] = 0

        dvals = sqrt(vals)
        nonzero_dvals = dvals!=0
        dvals[nonzero_dvals] = 1.0/dvals[nonzero_dvals]
        X = dot(U*dvals,U.conj().transpose())

        return dot(new_tensor,X).reshape(old_shape).transpose(old_indices)
# }}}
def normalizeAndReturnInverseNormalizer(tensor,index): # {{{
    """\
Like :func:`normalize`, but returns an additional matrix that if ``A, B =
normalizeAndReturnInverseNormalizer(tensor,index)`` then ``A * B`` (along
*index* of A) is equal to *tensor*.\
"""
    new_indices = list(range(tensor.ndim))
    del new_indices[index]
    new_indices.append(index)

    old_shape = list(tensor.shape)
    del old_shape[index]
    new_shape = (prod(old_shape),tensor.shape[index])
    old_shape.append(tensor.shape[index])

    new_tensor = tensor.transpose(new_indices).reshape(new_shape)

    old_indices = list(range(tensor.ndim-1))
    old_indices.insert(index,tensor.ndim-1)

    try:
        u, s, v = svd(new_tensor,full_matrices=0)
        return dot(u,v).reshape(old_shape).transpose(old_indices), dot(v.transpose().conj()*s,v)
    except LinAlgError:
        M = dot(new_tensor.conj().transpose(),new_tensor)

        vals, U = eigh(M)
        vals[vals<0] = 0

        dvals = sqrt(vals)
        nonzero_dvals = dvals!=0
        dvals[nonzero_dvals] = 1.0/dvals[nonzero_dvals]

        return dot(new_tensor,dot(U*dvals,U.conj().transpose())).reshape(old_shape).transpose(old_indices), dot(U*vals,U.conj().transpose())
# }}}
def normalizeAndDenormalize(tensor_to_normalize,index_to_normalize,tensor_to_denormalize,index_to_denormalize): # {{{
    """\
Let ``C, D = normalizeAndDenormalize(A,AI,B,BI)``; then ``C`` is normalized
(see :func:`normalize`) along axis ``AI`` and ``C * D`` (along ``AI`` and
``BI``) equals ``A * B``.\
"""
# }}}
def randomComplexSample(shape): # {{{
    """Returns a random sample of complex numbers with the given *shape*."""
    return random_sample(shape)*2-1+random_sample(shape)*2j-1j
# }}}
def replaceAt(iterable,index,new_value): # {{{
    """\
Replaces the value in *iterable* at *index* with *new_value*.

If the type of iterable can be called to create a new value with that type then
the value returned will have the same type as *iterable*; otherwise, the value
returned will be a tuple.\
"""
    new_values = (old_value if i != index else new_value for (i,old_value) in enumerate(iterable))
    try:
        return type(iterable)(new_values)
    except TypeError:
        return tuple(new_values)
# }}}
def relaxOver(initial,expectation_multiplier,normalization_multiplier=None,maximum_number_of_multiplications=None,tolerance=1e-7,dimension_of_krylov_space=3): # {{{
    """\
Computes the eigenvector corresponding to the least (most negative) eigenvalue of the linear operator given by the :class:`Multiplier` given by *expectation_muliplier* using *initial* as the initial guess.

initial
    the initial guess
expectation_multiplier
    the :class:`Multiplier` for the input matrix to the eigensolver
normalization_multiplier
    if given, specifies the normalization matrix for the generalized eigenvalue
    problem
maximum_number_of_multiplications
    if given, provides a cap on the number of multiplications to perform; if
    this cap is reached, then the solver gives up and returns the best result
    found so far
tolerance
    if the distance between the eigenvalues of two successive iterations is
    below this amount then the eigenvalue is declared to be converged
dimension_of_krylov_space
    if given, specifies the size of the Krylov space to use\
"""
    DataClass = type(initial)
    shape = initial.shape
    initial = initial.toArray().ravel()
    initial /= norm(initial)
    N = len(initial)

    if normalization_multiplier is None:
        applyInverseNormalization = lambda x: x
    elif normalization_multiplier.isCheaperToFormMatrix(10*2*dimension_of_krylov_space):
        applyInverseNormalization = partial(lu_solve,lu_factor(normalization_multiplier.formMatrix().toArray()))
        del normalization_multiplier
    else:
        normalization_matvec = lambda v: normalization_multiplier(DataClass(v.reshape(shape))).toArray().ravel()
        normalization_operator = LinearOperator(matvec=normalization_matvec,shape=(N,N),dtype=initial.dtype)
        def applyInverseNormalization(in_v):
            out_v, info = gmres(normalization_operator,in_v)
            assert info == 0
            return out_v

    if expectation_multiplier.isCheaperToFormMatrix(2*dimension_of_krylov_space):
        expectation_matrix = expectation_multiplier.formMatrix().toArray()
        multiplyExpectation = lambda v: dot(expectation_matrix,v)
        del expectation_multiplier
    else:
        multiplyExpectation = lambda v: expectation_multiplier(DataClass(v.reshape(shape))).toArray().ravel()

    multiply = lambda v: applyInverseNormalization(multiplyExpectation(v))
    initial_value = dot(initial.conj(),multiply(initial))

    number_of_multiplications = 0
    last_lowest_eigenvalue = None
    space_is_complete = dimension_of_krylov_space == N
    while True:
        krylov_basis = zeros((dimension_of_krylov_space,N),dtype=complex128)
        multiplied_krylov_basis = zeros((dimension_of_krylov_space,N),dtype=complex128)
        krylov_basis[0] = initial
        del initial
        for i in range(0,dimension_of_krylov_space):
            multiplied_krylov_basis[i] = multiply(krylov_basis[i])
            if i < dimension_of_krylov_space-1:
                # Note:  Subtracting and then normalizing is numerically unstable
                #        when the vector is close to being an eigenvector
                #        due to the loss of precision when taking the difference
                #        between the original vector and the multiplied vector
                krylov_basis[i+1] = multiplied_krylov_basis[i] - dot(dot(krylov_basis[:i+1].conj(),multiplied_krylov_basis[i]),krylov_basis[:i+1])
                normalization = norm(krylov_basis[i+1])
                if normalization <= 1e-14:
                    space_is_complete = True
                    krylov_basis = krylov_basis[:i+1]
                    multiplied_krylov_basis = multiplied_krylov_basis[:i+1]
                    break
                krylov_basis[i+1] /= normalization
        number_of_multiplications += dimension_of_krylov_space

        matrix_in_krylov_subspace = dot(krylov_basis.conj(),multiplied_krylov_basis.transpose())
        evals, evecs = eig(matrix_in_krylov_subspace)
        mindex = argmin(evals.real)
        mineval = evals[mindex]
        minevec = evecs[:,mindex]
        if space_is_complete or last_lowest_eigenvalue is not None and (abs(last_lowest_eigenvalue-mineval)<=tolerance) or maximum_number_of_multiplications is not None and number_of_multiplications >= maximum_number_of_multiplications:
            final = dot(minevec,krylov_basis)
            final /= norm(final)
            final_value = dot(final.conj(),multiply(final))
            if (final_value-initial_value)/(abs(final_value)+abs(initial_value)) > 1 + 1e-7:
                raise RelaxFailed(initial_value,final_value)
            return DataClass(final.reshape(shape))
        else:
            initial = dot(minevec,krylov_basis)
            initial /= norm(initial)
            last_lowest_eigenvalue = mineval
    # }}}
def unitize(matrix): # {{{
    """\
Finds the best unitary approximation to *matrix* by computing the SVD and then
setting all of the singular values to 1.\
"""
    U, _, V = svd(matrix,full_matrices=False)
    return dot(U,V)
# }}}
# }}}

# Index functions {{{
def O(i): # {{{
    """Computes the index of the side opposite to that of the given index."""
    return (i+2)%4
# }}}
def L(i): # {{{
    """Computes the index of the side left (counter-clockwise) of that of the given index."""
    return (i+1)%4
# }}}
def R(i): # {{{
    """Computes the index of the side right (clockwise) of that of the given index."""
    return (i-1)%4
# }}}
def A(d,i): # {{{
    """\
Returns *i*-1 if *i* > *d* and *i* otherwise.

This function is used when one of the indices has been deleted and so all
indices after that one are shifted down by one.\
"""
    return i-1 if i > d else i
# }}}
def OA(i): # {{{
    """\
Like :func:`O`, but shifts the resulting index down if it is after *i*,
intended for the case when the index *i* has been deleted.
"""
    return A(i,O(i))
# }}}
def LA(i): # {{{
    """\
Like :func:`L`, but shifts the resulting index down if it is after *i*,
intended for the case when the index *i* has been deleted.
"""
    return A(i,L(i))
# }}}
def RA(i): # {{{
    """\
Like :func:`R`, but shifts the resulting index down if it is after *i*,
intended for the case when the index *i* has been deleted.
"""
    return A(i,R(i))
# }}}
# }}}

# Exports {{{
__all__ = [
    "DimensionMismatchError",
    "InvariantViolatedError",
    "RelaxFailed",
    "UnexpectedTensorRankError",

    "Join",
    "Multiplier",

    "prepend",
    "prependDataContractor",

    "Pauli",

    "applyIndexMapTo",
    "applyPermutation",
    "buildProductTensor",
    "buildTensor",
    "checkForNaNsIn",
    "computeAndCheckNewDimension",
    "computeCompressor",
    "computeCompressorForMatrixTimesItsDagger",
    "computeAbsoluteLimitingLinearCoefficient",
    "computeNewDimension",
    "crand",
    "dropAt",
    "formDataContractor",
    "invertPermutation",
    "maximumNewBandwidth",
    "multiplyTensorByMatrixAtIndex",
    "normalize",
    "normalizeAndReturnInverseNormalizer",
    "normalizeAndDenormalize",
    "randomComplexSample",
    "replaceAt",
    "relaxOver",
    "unitize",

    "O",
    "L",
    "R",
    "A",
    "OA",
    "LA",
    "RA",
]
# }}}
