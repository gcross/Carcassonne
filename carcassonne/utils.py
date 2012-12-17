# Imports {{{
from collections import defaultdict
from functools import partial, reduce
from numpy import argsort, argmax, array, complex128, dot, identity, multiply, prod, sqrt, set_printoptions, tensordot, trace, zeros
from numpy.random import rand, random_sample
from scipy.linalg import LinAlgError, eig, eigh, eigvals, lu_factor, lu_solve, norm, svd, qr
from scipy.sparse.linalg import LinearOperator, eigs, eigsh, gmres
# }}}

set_printoptions(linewidth=132)

# Exceptions {{{
class DimensionMismatchError(ValueError): # {{{
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
class InvariantViolatedError(Exception): pass
class UnexpectedTensorRankError(ValueError): # {{{
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
        assert self.left_tensor_number == other.left_tensor_number
        assert self.right_tensor_number == other.right_tensor_number
        self.left_tensor_indices  += other.left_tensor_indices
        self.right_tensor_indices += other.right_tensor_indices
    # }}}
    def update(self,old_tensor_number,new_tensor_number,index_map): # {{{
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
class FromLeft: # {{{
    # Constants {{{
    number_of_left_dimensions = 1
    number_of_right_dimensions = 0
    # }}}
    # Properties {{{
    left_dimensions = property(lambda self: (self.dimension,))
    right_dimensions = property(lambda self: ())
    # }}}
    def __init__(self,dimension): # {{{
        self.dimension = dimension
    # }}}
    def appendDimensionsToTransposition(self,transposition,left_transpose_offsets,right_transpose_offsets): # {{{
        transposition.append(left_transpose_offsets[self.dimension])
    # }}}
    def getResultDimension(self,left_dimensions,right_dimensions): # {{{
        return left_dimensions[self.dimension]
    # }}}
    def getResultIndex(self,right_dimensions,left_indices,right_indices): # {{{
        return left_indices[self.dimension]
    # }}}
# }}}
class FromRight: # {{{
    # Constants {{{
    number_of_left_dimensions = 0
    number_of_right_dimensions = 1
    # }}}
    # Properties {{{
    left_dimensions = property(lambda self: ())
    right_dimensions = property(lambda self: (self.dimension,))
    # }}}
    def __init__(self,dimension): # {{{
        self.dimension = dimension
    # }}}
    def appendDimensionsToTransposition(self,transposition,left_transpose_offsets,right_transpose_offsets): # {{{
        transposition.append(right_transpose_offsets[self.dimension])
    # }}}
    def getResultDimension(self,left_dimensions,right_dimensions): # {{{
        return right_dimensions[self.dimension]
    # }}}
    def getResultIndex(self,right_dimensions,left_indices,right_indices): # {{{
        return right_indices[self.dimension]
    # }}}
# }}}
class FromBoth: # {{{
    # Constants {{{
    number_of_left_dimensions = 1
    number_of_right_dimensions = 1
    # }}}
    # Properties {{{
    left_dimensions = property(lambda self: (self.left_dimension,))
    right_dimensions = property(lambda self: (self.right_dimension,))
    # }}}
    def __init__(self,left_dimension,right_dimension,indices_to_ignore=frozenset(),indices_to_redirect=frozenset()): # {{{
        self.left_dimension = left_dimension
        self.right_dimension = right_dimension
        self.indices_to_ignore = indices_to_ignore
        self.indices_to_redirect = indices_to_redirect
    # }}}
    def appendDimensionsToTransposition(self,transposition,left_transpose_offsets,right_transpose_offsets): # {{{
        transposition.append(left_transpose_offsets[self.left_dimension])
        transposition.append(right_transpose_offsets[self.right_dimension])
    # }}}
    def getResultDimension(self,left_dimensions,right_dimensions): # {{{
        return left_dimensions[self.left_dimension]*right_dimensions[self.right_dimension]
    # }}}
    def getResultIndex(self,right_dimensions,left_indices,right_indices): # {{{
        left_index = left_indices[self.left_dimension]
        right_index = right_indices[self.right_dimension]
        indices = (left_index,right_index)
        if indices in self.indices_to_ignore:
            return None
        if indices in self.indices_to_redirect:
            return self.indices_to_redirect[indices]
        return left_index * right_dimensions[self.right_dimension] + right_index
    # }}}
# }}}
class Multiplier: # {{{
    def __init__(self,shape,multiply,cost_of_multiply,formMatrix,cost_of_formMatrix): # {{{
        self.shape = shape
        self.multiply = multiply
        self.cost_of_multiply = cost_of_multiply
        self.formMatrix = formMatrix
        self.cost_of_formMatrix = cost_of_formMatrix
    # }}}
    def __call__(self,vector): # {{{
        return self.multiply(vector)
    # }}}
    @classmethod # fromMatrix {{{
    def fromMatrix(self,matrix):
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
        return estimated_number_of_applications*self.cost_of_multiply > \
                self.cost_of_formMatrix + estimated_number_of_applications*self.shape[0]*self.shape[1]
    # }}}
# }}}
# }}}

# Decorators {{{
class prepend: # {{{
    def __init__(self,*args,**keywords):
        self.args = args
        self.keywords = keywords
    def __call__(self,f):
        return partial(f,*self.args,**self.keywords)
# }}}
def prependContractor(*args,**keywords): # {{{
    return prepend(formContractor(*args,**keywords))
# }}}
def prependDataContractor(*args,**keywords): # {{{
    return prepend(formDataContractor(*args,**keywords))
# }}}
# }}}

# Pauli Operators {{{
class Pauli:
    I = identity(2,dtype=complex128)
    X = array([[0,1],[1,0]],dtype=complex128)
    Y = array([[0,-1j],[1j,0]],dtype=complex128)
    Z = array([[1,0],[0,-1]],dtype=complex128)
# }}}

# Functions {{{
def applyIndexMapTo(index_map,indices): # {{{
    return [index_map[index] for index in indices]
# }}}
def applyPermutation(permutation,values): # {{{
    return [values[i] for i in permutation]
# }}}
def buildProductTensor(*factors): # {{{
    return reduce(multiply.outer,(array(factor,dtype=complex128) for factor in factors)) #,zeros((),dtype=complex128))
# }}}
def buildTensor(dimensions,components): # {{{
    tensor = zeros(dimensions,dtype=complex128)
    for index, value in components.items():
        tensor[index] = value
    return tensor
# }}}
def checkForNaNsIn(data): # {{{
    assert not data.hasNaN()
    return data
# }}}
def crand(*shape): # {{{
    return rand(*shape)*2-1+rand(*shape)*2j-1j
# }}}
def computeAndCheckNewDimension(state_center,direction,by=None,to=None,do_as_much_as_possible=False): # {{{
    old_dimension = state_center.shape[direction]
    new_dimension = computeNewDimension(old_dimension,by=by,to=to)
    increment = new_dimension-old_dimension
    maximum_increment = maximumBandwidthIncrement(direction,state_center.shape)
    if increment > maximum_increment:
        if do_as_much_as_possible:
            increment = maximum_increment
            new_dimension = old_dimension + increment
        else:
            raise ValueError("Increment of {} in the bandwidth dimensions {} and {} is too great given the current shape of {} (maximum increment is {}).".format(increment,direction,direction+2,state_center_data.shape,maximum_increment))
    return old_dimension, new_dimension, increment
# }}}
def computeCompressor(old_dimension,new_dimension,multiplier,dtype,normalize=False): # {{{
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
    if normalize:
        evals = sqrt(evals).reshape(new_dimension,1)
        compressor = evecs * evals
        inverse_compressor_conj = evecs / evals
    else:
        compressor = evecs
        inverse_compressor_conj = evecs
    return compressor, inverse_compressor_conj
# }}}
def computeCompressorForMatrixTimesItsDagger(old_dimension,new_dimension,matrix,normalize=False): # {{{
    other_dimension = matrix.shape[0]
    matrix_dagger = matrix.transpose().conj()
    return \
        computeCompressor(
            old_dimension,
            new_dimension,
            Multiplier(
                (old_dimension,)*2,
                lambda v: dot(matrix_dagger,dot(matrix,v)),
                2 * old_dimension * other_dimension,
                lambda: dot(matrix_dagger,matrix),
                old_dimension**2 * other_dimension
            ),
            matrix.dtype,
            normalize
        )
# }}}
def computeLengthAndCheckForGaps(indices,error_message): # {{{
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
def computeLimitingLinearCoefficient(n,multiplyO,multiplyN,multiplyL,multiplyR): # {{{
    if n <= 3:
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
        ovecs = eigs(LinearOperator((n,n),matvec=multiplyO),k=2,which='LM',ncv=9)[1].transpose()

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
def computeNormalizerAndInverse(matrix,index): # {{{
    new_indices = list(range(matrix.ndim))
    del new_indices[index]
    new_indices.append(index)

    size_of_normalization_dimension = matrix.shape[index]

    old_shape = list(matrix.shape)
    del old_shape[index]
    new_shape = (prod(old_shape),size_of_normalization_dimension)
    old_shape.append(size_of_normalization_dimension)

    new_matrix = matrix.transpose(new_indices).reshape(new_shape)

    old_indices = list(range(matrix.ndim-1))
    old_indices.insert(index,matrix.ndim-1)

    try:
        u, s, v = svd(new_matrix,full_matrices=0)
        return dot(v.transpose().conj()*(1/s),v), dot(v.transpose().conj()*s,v)
    except LinAlgError:
        M = dot(new_matrix.conj().transpose(),new_matrix)

        vals, U = eigh(M)
        vals[vals<0] = 0

        dvals = sqrt(vals)
        nonzero_dvals = dvals!=0
        dvals[nonzero_dvals] = 1.0/dvals[nonzero_dvals]

        return dot(U*dvals,U.conj().transpose()), dot(U*vals,U.conj().transpose())
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
    new_values = (x for i, x in enumerate(iterable) if i != index)
    try:
        return type(iterable)(new_values)
    except TypeError:
        return tuple(new_values)
# }}}
def formAbsorber(left_join_dimensions,right_join_dimensions,result_dimension_sources): # {{{
    # Compute the numbers of dimensions {{{
    number_of_join_dimensions = len(left_join_dimensions)
    assert number_of_join_dimensions == len(right_join_dimensions)
    number_of_left_dimensions = number_of_join_dimensions + sum(x.number_of_left_dimensions for x in result_dimension_sources)
    number_of_right_dimensions = number_of_join_dimensions + sum(x.number_of_right_dimensions for x in result_dimension_sources)
    # }}}
    # Compute the transposition {{{
    next_transpose_dimension = 0

    left_transpose_offsets = {}
    for dimension in range(number_of_left_dimensions):
        if dimension not in left_join_dimensions:
            left_transpose_offsets[dimension] = next_transpose_dimension
            next_transpose_dimension += 1

    right_transpose_offsets = {}
    for dimension in range(number_of_right_dimensions):
        if dimension not in right_join_dimensions:
            right_transpose_offsets[dimension] = next_transpose_dimension
            next_transpose_dimension += 1

    transposition = []
    for result_dimension_source in result_dimension_sources:
        result_dimension_source.appendDimensionsToTransposition(transposition,left_transpose_offsets,right_transpose_offsets)
    transposition = tuple(transposition)
    # }}}
    join_dimensions = (left_join_dimensions,right_join_dimensions)
    def absorb(left,right): # {{{
        left_shape = left.shape
        right_shape = right.shape
        result_shape = tuple(x.getResultDimension(left_shape,right_shape) for x in result_dimension_sources)
        return tensordot(left,right,join_dimensions).transpose(transposition).reshape(result_shape)
    # }}}
    return absorb
# }}}
def formContractor(order,joins,result_joins): # {{{
    observed_tensor_indices = {}

    # Tabulate all of the observed tensor indices {{{
    for (tensor_id,index) in sum([list(x) for x in joins] + [list(x) for x in result_joins],[]):
        if index < 0:
            raise ValueError("index {} of tensor {} is negative".format(index,tensor_id))
        try:
            observed_indices = observed_tensor_indices[tensor_id]
        except KeyError:
            observed_indices = set()
            observed_tensor_indices[tensor_id] = observed_indices
        if index in observed_indices:
            raise ValueError("index {} of tensor {} appears more than once in the joins".format(index,tensor_id))
        observed_indices.add(index)

    for tensor_id in observed_tensor_indices:
        if tensor_id not in order:
            raise ValueError("tensor {} does not appear in the list of arguments ({})".format(tensor_id,order))
    # }}}

    tensor_join_ids = {}

    # Check the observed tensor indices and initialize the map of tensor joins {{{
    for (tensor_id,observed_indices) in observed_tensor_indices.items():
        tensor_dimension = max(observed_indices)+1
        expected_indices = set(range(tensor_dimension))
        invalid_indices = observed_indices - expected_indices
        missing_indices = expected_indices - observed_indices
        if len(invalid_indices) > 0:
            raise ValueError('the invalid indices {} have appeared in joins involving tendor {}'.format(invalid_indices,tensor_id))
        if len(missing_indices) > 0:
            raise ValueError('the expected indices {} do not appear in any of the joins for tensor {}'.format(missing_indices,tensor_id))
        if tensor_id not in order:
            raise ValueError('tensor {} does not appear in the list of arguments'.format(tensor_id))
        tensor_join_ids[tensor_id] = [None]*tensor_dimension
    # }}}

    result_join_ids = []

    # Label each join with a unique id {{{
    current_join_id = 0
    for join in joins:
        for (tensor_id,index) in join:
            tensor_join_ids[tensor_id][index] = current_join_id
        current_join_id += 1
    for join in result_joins:
        join_ids = []
        for (tensor_id,index) in join:
            join_ids.append(current_join_id)
            tensor_join_ids[tensor_id][index] = current_join_id
            current_join_id += 1
        result_join_ids.append(join_ids)
    # }}}

    argument_join_ids = [tensor_join_ids[tensor_id] for tensor_id in order]

    # Form the contractor function {{{
    def contract(*arguments):
        if len(arguments) != len(order):
            raise ValueError("wrong number of arguments;  expected {} but received {}".format(len(order),len(arguments)))
        for (i, (tensor_id, argument)) in enumerate(zip(order,arguments)):
            if argument.ndim != len(tensor_join_ids[tensor_id]):
                raise ValueError("argument {} ('{}') has rank {} when it was expected to have rank {}".format(i,order[i],argument.ndim,len(tensor_join_ids[tensor_id])))
        arguments = list(arguments)
        join_ids_index = -1
        current_tensor = arguments.pop()
        current_join_ids = argument_join_ids[join_ids_index]
        while len(arguments) > 0:
            join_ids_index -= 1
            next_tensor = arguments.pop()
            next_join_ids = argument_join_ids[join_ids_index]
            try:
                first_axes = []
                second_axes = []
                first_axis_index = 0
                common_join_ids = set()
                for join_id in current_join_ids:
                    if join_id in next_join_ids:
                        common_join_ids.add(join_id)
                        first_axes.append(first_axis_index)
                        second_axes.append(next_join_ids.index(join_id))
                    first_axis_index += 1
                current_tensor = tensordot(current_tensor,next_tensor,(first_axes,second_axes))
                current_join_ids = [i for i in current_join_ids+next_join_ids if i not in common_join_ids]
            except Exception as e:
                raise ValueError("Error when joining tensor {}: '{}' (first tensor axes are {} with dimensions {}, second ({}) axes are {} with dimensions {})".format(order[join_ids_index],str(e),first_axes,[current_tensor.shape[i] for i in first_axes],order[join_ids_index],second_axes,[next_tensor.shape[i] for i in second_axes]))
        current_tensor = current_tensor.transpose([current_join_ids.index(i) for i in sum([list(x) for x in result_join_ids],[])])
        old_shape = current_tensor.shape
        new_shape = []
        index = 0
        for join in result_join_ids:
            dimension = 1
            for _ in join:
                dimension *= old_shape[index]
                index += 1
            new_shape.append(dimension)
        return current_tensor.reshape(new_shape)
    # }}}

    return contract
# }}}
def formDataContractor(joins,final_groups,tensor_ranks=None): # {{{
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
    return [permutation.index(i) for i in range(len(permutation))]
# }}}
def maximumBandwidthIncrement(direction,shape): # {{{
    return min(shape[direction],prod(dropAt(shape,direction)))
# }}}
def multiplyBySingleSiteOperator(state,operator): # {{{
    return state.absorbMatrixAt(4,operator)
# }}}
def multiplyTensorByMatrixAtIndex(tensor,matrix,index): # {{{
    tensor_new_indices = list(range(tensor.ndim-1))
    tensor_new_indices.insert(index,tensor.ndim-1)
    return tensordot(tensor,matrix,(index,0)).transpose(tensor_new_indices)
# }}}
def normalize(matrix,index): # {{{
    new_indices = list(range(matrix.ndim))
    del new_indices[index]
    new_indices.append(index)

    old_shape = list(matrix.shape)
    del old_shape[index]
    new_shape = (prod(old_shape),matrix.shape[index])
    old_shape.append(matrix.shape[index])

    new_matrix = matrix.transpose(new_indices).reshape(new_shape)
    if new_matrix.shape[1] > new_matrix.shape[0]:
        raise ValueError("There are not enough degrees of freedom available to normalize the tensor.")

    old_indices = list(range(matrix.ndim-1))
    old_indices.insert(index,matrix.ndim-1)

    try:
        u, s, v = svd(new_matrix,full_matrices=0)
        return dot(u,v).reshape(old_shape).transpose(old_indices)
    except LinAlgError:
        M = dot(new_matrix.conj().transpose(),new_matrix)

        vals, U = eigh(M)
        vals[vals<0] = 0

        dvals = sqrt(vals)
        nonzero_dvals = dvals!=0
        dvals[nonzero_dvals] = 1.0/dvals[nonzero_dvals]
        X = dot(U*dvals,U.conj().transpose())

        return dot(new_matrix,X).reshape(old_shape).transpose(old_indices)
# }}}
def normalizeAndReturnInverseNormalizer(matrix,index): # {{{
    new_indices = list(range(matrix.ndim))
    del new_indices[index]
    new_indices.append(index)

    old_shape = list(matrix.shape)
    del old_shape[index]
    new_shape = (prod(old_shape),matrix.shape[index])
    old_shape.append(matrix.shape[index])

    new_matrix = matrix.transpose(new_indices).reshape(new_shape)

    old_indices = list(range(matrix.ndim-1))
    old_indices.insert(index,matrix.ndim-1)

    try:
        u, s, v = svd(new_matrix,full_matrices=0)
        return dot(u,v).reshape(old_shape).transpose(old_indices), dot(v.transpose().conj()*s,v)
    except LinAlgError:
        M = dot(new_matrix.conj().transpose(),new_matrix)

        vals, U = eigh(M)
        vals[vals<0] = 0

        dvals = sqrt(vals)
        nonzero_dvals = dvals!=0
        dvals[nonzero_dvals] = 1.0/dvals[nonzero_dvals]

        return dot(new_matrix,dot(U*dvals,U.conj().transpose())).reshape(old_shape).transpose(old_indices), dot(U*vals,U.conj().transpose())
# }}}
def normalizeAndDenormalize(tensor_to_normalize,index_to_normalize,tensor_to_denormalize,index_to_denormalize): # {{{
    normalized_tensor, inverse_normalizer = normalizeAndReturnInverseNormalizer(tensor_to_normalize,index_to_normalize)
    unnormalized_tensor = multiplyTensorByMatrixAtIndex(tensor_to_denormalize,inverse_normalizer.transpose(),index_to_denormalize)
    return normalized_tensor, unnormalized_tensor
# }}}
def randomComplexSample(shape): # {{{
    return random_sample(shape)*2-1+random_sample(shape)*2j-1j
# }}}
def replaceAt(iterable,index,new_value): # {{{
    new_values = (old_value if i != index else new_value for (i,old_value) in enumerate(iterable))
    try:
        return type(iterable)(new_values)
    except TypeError:
        return tuple(new_values)
# }}}
def relaxOver(initial,expectation_multiplier,normalization_multiplier=None,maximum_number_of_multiplications=None,tolerance=1e-8,dimension_of_krylov_space=None): # {{{
    DataClass = type(initial)
    shape = initial.shape
    initial = initial.toArray().ravel()
    initial /= norm(initial)
    N = len(initial)
    if dimension_of_krylov_space is None:
        dimension_of_krylov_space = 3

        if normalization_multiplier is None:
            applyInverseNormalization = lambda x: x
        elif normalization_multiplier.isCheaperToFormMatrix(10*2*dimension_of_krylov_space):
            applyInverseNormalization = partial(lu_solve,lu_factor(normalization_multiplier.formMatrix().toArray()))
            del normalization_multiplier
        else:
            normalization_matvec = lambda v: normalization_multiplier(DataClass(v.reshape(self.shape))).toArray().ravel()
            normalization_operator = LinearOperator(matvec=normalization_matvec,shape=(N,N),dtype=self.dtype)
            def applyInverseNormalization(in_v):
                out_v, info = gmres(normalization_operator,in_v)
                assert info == 0
                return out_v

        if expectation_multiplier.isCheaperToFormMatrix(2*dimension_of_krylov_space):
            expectation_matrix = expectation_multiplier.formMatrix().toArray()
            multiplyExpectation = lambda v: dot(expectation_matrix,v)
            del expectation_multiplier
        else:
            multiplyExpectation = lambda v: expectation_multiplier(DataClass(v.reshape(self.shape))).toArray().ravel()

        multiply = lambda v: applyInverseNormalization(multiplyExpectation(v))

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
            evecs = evecs.transpose()
            permutation = argsort(evals)
            evals = evals[permutation]
            evecs = evecs[permutation]
            if space_is_complete or last_lowest_eigenvalue is not None and (abs(last_lowest_eigenvalue-evals[0])<=tolerance) or maximum_number_of_multiplications is not None and number_of_multiplications >= maximum_number_of_multiplications:
                return DataClass(dot(evecs[0],krylov_basis).reshape(shape))
            else:
                initial = dot(evecs[0],krylov_basis)
                initial /= norm(initial)
                last_lowest_eigenvalue = evals[0]
    # }}}
# }}}

# Index functions {{{
def O(i): return (i+2)%4
def L(i): return (i+1)%4
def R(i): return (i-1)%4
def A(d,i): return i-1 if i > d else i
def OA(i): return A(i,O(i))
def LA(i): return A(i,L(i))
def RA(i): return A(i,R(i))
# }}}

# Exports {{{
__all__ = [
    "DimensionMismatchError",
    "InvariantViolatedError",
    "UnexpectedTensorRankError",

    "FromLeft",
    "FromRight",
    "FromBoth",
    "Join",
    "Multiplier",

    "prepend",
    "prependContractor",
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
    "computeLimitingLinearCoefficient",
    "computeNewDimension",
    "computeNormalizerAndInverse",
    "crand",
    "dropAt",
    "formAbsorber",
    "formContractor",
    "formDataContractor",
    "invertPermutation",
    "maximumBandwidthIncrement",
    "multiplyTensorByMatrixAtIndex",
    "normalize",
    "normalizeAndReturnInverseNormalizer",
    "normalizeAndDenormalize",
    "randomComplexSample",
    "replaceAt",
    "relaxOver",

    "O",
    "L",
    "R",
    "A",
    "OA",
    "LA",
    "RA",
]
# }}}
