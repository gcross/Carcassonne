# Imports {{{
from collections import defaultdict
from functools import partial
from numpy import sqrt, tensordot, zeros
from numpy.linalg import eigh
from numpy.random import rand, random_sample
from scipy.sparse.linalg import LinearOperator, eigs, eigsh
# }}}

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
# }}}

# Decorators {{{
class prepend: # {{{
    def __init__(self,*args,**keywords):
        self.args = args
        self.keywords = keywords
    def __call__(self,f):
        return partial(f,*self.args,**self.keywords)
# }}}
def prependDataContractor(*args,**keywords): # {{{
    return prepend(formDataContractor(*args,**keywords))
# }}}
# }}}

# Functions {{{
def applyIndexMapTo(index_map,indices): # {{{
    return [index_map[index] for index in indices]
# }}}
def applyPermutation(permutation,values): # {{{
    return [values[i] for i in permutation]
# }}}
def checkForNaNsIn(data): # {{{
    assert not data.hasNaN()
    return data
# }}}
def crand(*shape): # {{{
    return rand(*shape)*2-1+rand(*shape)*2j-1j
# }}}
def computeCompressor(old_dimension,new_dimension,matvec,dtype,computeDenseMatrix,normalize=False): # {{{
    if new_dimension < 0:
        raise ValueError("New dimension ({}) must be non-negative.".format(new_dimension))
    elif new_dimension > old_dimension:
        raise ValueError("New dimension ({}) must be less than or equal to the old dimension ({}).".format(new_dimension,old_dimension))
    elif old_dimension == 0:
        return (zeros((new_dimension,old_dimension),dtype=dtype),)*2
    elif new_dimension >= old_dimension // 2:
        evals, evecs = eigh(computeDenseMatrix())
        evals = evals[-new_dimension:]
        evecs = evecs[:,-new_dimension:]
    else:
        operator = \
            LinearOperator(
                shape=(old_dimension,)*2,
                matvec=matvec,
                dtype=dtype
            )
        evals, evecs = eigsh(operator,k=new_dimension)
    evecs = evecs.transpose()
    while abs(evals[new_dimension-1]) < 1e-15:
        new_dimension -= 1
    if normalize:
        evals = sqrt(evals).reshape(new_dimension,1)
        compressor = evecs * evals
        inverse_compressor_conj = evecs / evals
    else:
        compressor = evecs
        inverse_compressor_conj = evecs
    return compressor, inverse_compressor_conj
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
    return type(iterable)(x for i, x in enumerate(iterable) if i != index)
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
def multiplyBySingleSiteOperator(state,operator): # {{{
    return state.absorbMatrixAt(4,operator)
# }}}
def randomComplexSample(shape): # {{{
    return random_sample(shape)*2-1+random_sample(shape)*2j-1j
# }}}
def replaceAt(iterable,index,new_value): # {{{
    return type(iterable)(old_value if i != index else new_value for (i,old_value) in enumerate(iterable))
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
    "UnexpectedTensorRankError",

    "FromLeft",
    "FromRight",
    "FromBoth",
    "Join",

    "prepend",
    "prependDataContractor",

    "applyIndexMapTo",
    "applyPermutation",
    "checkForNaNsIn",
    "computeCompressor",
    "computeNewDimension",
    "crand",
    "dropAt",
    "formAbsorber",
    "formContractor",
    "formDataContractor",
    "invertPermutation",
    "randomComplexSample",
    "replaceAt",

    "O",
    "L",
    "R",
    "A",
    "OA",
    "LA",
    "RA",
]
# }}}
