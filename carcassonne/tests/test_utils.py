# Imports {{{
from functools import reduce
from numpy import dot, multiply, prod
from paycheck import *
from scipy.linalg import LinAlgError, eigh, eigvalsh, svd
from scipy.sparse.linalg import eigs, eigsh

from ..data import NDArrayData
from ..utils import *
from . import *
# }}}

# Helper classes {{{

class TaggedNDArrayData(NDArrayData): # {{{
    def __init__(self,arr,tag):
        NDArrayData.__init__(self,arr)
        self.tag = tag
    def contractWith(self,other,self_axes,other_axes):
        return TaggedNDArrayData(NDArrayData.contractWith(self,other,self_axes,other_axes).toArray(),(self.tag,other.tag))
    def join(self,*groups):
        return TaggedNDArrayData(NDArrayData.join(self,*groups).toArray(),self.tag)
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

# Helper functions {{{
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
# }}}

class TestFormDataContractor(TestCase): # {{{
    @with_checker
    def test_single_tensor_outer_product(self,ndim=irange(1,5)): # {{{
        data = NDArrayData.newRandom(*randomShape(ndim))
        contract = formDataContractor([],[[(0,i) for i in range(ndim)]])
        contracted_data = contract(data)
        raveled_data = data.join(*[range(ndim)])
        self.assertDataAlmostEqual(raveled_data,contracted_data)
    # }}}
    @with_checker
    def test_multiple_tensors_outer_product(self,number_of_tensors=irange(1,5)): # {{{
        tensors = [NDArrayData.newRandom(*randomShape(randint(1,3),3)) for _ in range(number_of_tensors)]
        contraction = formDataContractor([],[[(tensor_number,i)] for tensor_number, tensor in enumerate(tensors) for i in range(tensor.ndim)])(*tensors)
        self.assertDataAlmostEqual(
            reduce(lambda x,y: x.contractWith(y,[],[]),tensors),
            contraction
        )
        self.assertAllClose(
            reduce(lambda x,y: multiply.outer(x,y),[tensor.toArray() for tensor in tensors]),
            contraction.toArray()
        )
    # }}}
    @with_checker
    def test_two_tensors_inner_product_resulting_in_scalar(self,ndim=irange(1,5)): # {{{
        A_shape = randomShape(ndim)
        A = NDArrayData.newRandom(*A_shape)
        B_permutation = randomPermutation(ndim)
        B_inverse_permutation = invertPermutation(B_permutation)
        B_shape = applyPermutation(B_permutation,A_shape)
        B = NDArrayData.newRandom(*B_shape)
        axes = list(zip(range(ndim),B_inverse_permutation))
        shuffle(axes)
        A_axes, B_axes = zip(*axes)
        contraction = formDataContractor([Join(0,A_axes,1,B_axes)],[])(A,B)
        self.assertAlmostEqual(
            A.contractWith(B,A_axes,B_axes).extractScalar(),
            contraction
        )
        self.assertAlmostEqual(
            dot(A.toArray().ravel(),B.toArray().transpose(B_inverse_permutation).ravel()),
            contraction
        )
    # }}}
    @with_checker
    def test_two_tensors_inner_product_resulting_in_tensor(self, # {{{
        left_ndim=irange(1,3),
        middle_ndim=irange(1,3),
        right_ndim=irange(1,3),
    ):
        left_shape = randomShape(left_ndim,3)
        middle_shape = randomShape(middle_ndim,3)
        right_shape = randomShape(right_ndim,3)

        A_ndim = left_ndim + middle_ndim
        A_permutation = randomPermutation(A_ndim)
        A_shape = applyPermutation(A_permutation,left_shape + middle_shape)
        A_axes = applyPermutation(invertPermutation(A_permutation),range(A_ndim))
        A_left_axes, A_middle_axes = A_axes[:left_ndim], A_axes[left_ndim:]
        A = NDArrayData.newRandom(*A_shape)

        B_ndim = middle_ndim + right_ndim
        B_permutation = randomPermutation(B_ndim)
        B_shape = applyPermutation(B_permutation,middle_shape + right_shape)
        B_axes = applyPermutation(invertPermutation(B_permutation),range(B_ndim))
        B_middle_axes, B_right_axes = B_axes[:middle_ndim], B_axes[middle_ndim:]
        B = NDArrayData.newRandom(*B_shape)

        A_left_axes.sort()
        B_right_axes.sort()

        contraction = formDataContractor(
            [Join(0,A_middle_axes,1,B_middle_axes)],
            [[(tensor_number,axis)] for tensor_number, axes in enumerate([A_left_axes,B_right_axes]) for axis in axes]
        )(A,B)

        self.assertDataAlmostEqual(
            A.contractWith(B,A_middle_axes,B_middle_axes),
            contraction
        )
    # }}}
    @with_checker
    def test_matrix_multiplication_two_matrices(self,m=irange(1,10),k=irange(1,10),n=irange(1,10)): # {{{
        A = NDArrayData.newRandom(m,k)
        B = NDArrayData.newRandom(k,n)
        contraction = formDataContractor([Join(0,1,1,0)],[[(0,0)],[(1,1)]])(A,B)
        self.assertDataAlmostEqual(
            A.contractWith(B,[1],[0]),
            contraction
        )
        self.assertAllClose(
            dot(A.toArray(),B.toArray()),
            contraction.toArray()
        )
    # }}}
    @with_checker
    def test_matrix_multiplication_four_matrices_with_order_checking(self, # {{{
        a=irange(1,10),
        b=irange(1,10),
        c=irange(1,10),
        d=irange(1,10),
        e=irange(1,10),
    ):

        A = TaggedNDArrayData(crand(a,b),'A')
        B = TaggedNDArrayData(crand(b,c),'B')
        C = TaggedNDArrayData(crand(c,d),'C')
        D = TaggedNDArrayData(crand(d,e),'D')

        correct_value = reduce(dot,[tensor.toArray() for tensor in [A,B,C,D]])
        contraction = formDataContractor(
            [
                Join(0,1,1,0),
                Join(2,1,3,0),
                Join(1,1,2,0),
            ],
            [[(0,0)],[(3,1)]]
        )(A,B,C,D)
        self.assertAllClose(
            correct_value,
            contraction.toArray()
        )
        self.assertEqual(
            (('A','B'),('C','D')),
            contraction.tag
        )
    # }}}
    @with_checker
    def test_matrix_multiplication_three_matrices(self, # {{{
        a=irange(1,10),
        b=irange(1,10),
        c=irange(1,10),
        d=irange(1,10),
    ):
        A = NDArrayData.newRandom(a,b)
        B = NDArrayData.newRandom(b,c)
        C = NDArrayData.newRandom(c,d)
        correct_value = reduce(dot,[tensor.toArray() for tensor in [A,B,C]])
        joins = [
            Join(0,1,1,0),
            Join(1,1,2,0),
        ]
        shuffle(joins)
        contraction = formDataContractor(joins,[[(0,0)],[(2,1)]])(A,B,C)
        self.assertAllClose(
            correct_value,
            contraction.toArray()
        )
    # }}}
    @with_checker
    def test_matrix_multiplication_many_matrices(self, number_of_matrices=irange(1,10)): # {{{
        dimensions = [randint(1,10) for _ in range(number_of_matrices+1)]
        matrices = [NDArrayData.newRandom(*(dimensions[i],dimensions[i+1])) for i in range(number_of_matrices)]
        correct_value = reduce(dot,[matrix.toArray() for matrix in matrices])
        joins = [Join(i,1,i+1,0) for i in range(number_of_matrices-1)]
        shuffle(joins)
        contraction = formDataContractor(joins,[[(0,0)],[(number_of_matrices-1,1)]])(*matrices)
        self.assertAllClose(
            correct_value,
            contraction.toArray()
        )
    # }}}
    @with_checker
    def test_unexpected_tensor_rank_detected(self,number_of_tensors=irange(1,5)): # {{{
        ndims = [randint(1,5) for _ in range(number_of_tensors)]
        tensors = [NDArrayData.newRandom(*randomShape(ndim)) for ndim in ndims]
        broken_tensor_number = randint(0,len(ndims)-1)
        actual_rank = ndims[broken_tensor_number]
        expected_rank = actual_rank + 1
        ndims[broken_tensor_number] += 1
        try:
            formDataContractor([],[[(tensor_number,index)] for tensor_number in range(number_of_tensors) for index in range(ndims[tensor_number])])(*tensors)
            self.fail("exception was not thrown")
        except UnexpectedTensorRankError as e:
            self.assertEqual(e.tensor_number,broken_tensor_number)
            self.assertEqual(e.expected_rank,expected_rank)
            self.assertEqual(e.actual_rank,actual_rank)
    # }}}
    @with_checker
    def test_dimension_mismatch_detected(self,left_ndim=irange(1,5),right_ndim=irange(1,5)): # {{{
        left_tensor_number = randint(0,1)

        left_shared_shape = randomShape(left_ndim)
        left_permutation = randomPermutation(left_ndim)
        left_inverse_permutation = invertPermutation(left_permutation) 
        left_shape = applyPermutation(left_permutation,left_shared_shape)
        if left_tensor_number == 0:
            right_tensor_number = 1
            right_index = randint(0,left_ndim-1)
            right_dimension = left_shared_shape[right_index]
            left_index = left_inverse_permutation[right_index]
            left_shape[left_index] += 1
            left_dimension = left_shape[left_index]
        left_tensor = NDArrayData.newRandom(*left_shape)

        right_shared_shape = randomShape(right_ndim)
        right_permutation = randomPermutation(right_ndim)
        right_inverse_permutation = invertPermutation(right_permutation)
        right_shape = applyPermutation(right_permutation,right_shared_shape)
        if left_tensor_number == 1:
            right_tensor_number = 2
            left_index = randint(0,right_ndim-1)
            left_dimension = right_shared_shape[left_index]
            right_index = right_inverse_permutation[left_index]
            right_shape[right_index] += 1
            right_dimension = right_shape[right_index]
            left_index += left_ndim
        right_tensor = NDArrayData.newRandom(*right_shape)

        middle_tensor = NDArrayData.newRandom(*left_shared_shape + right_shared_shape)
        try:
            formDataContractor(
                [
                    Join(0,left_inverse_permutation,1,range(left_ndim)),
                    Join(2,right_inverse_permutation,1,range(left_ndim,left_ndim+right_ndim))
                ],
            [])(left_tensor,middle_tensor,right_tensor)
            self.fail("exception not thrown")
        except DimensionMismatchError as e:
            self.assertEqual(e.left_tensor_number,left_tensor_number)
            self.assertEqual(e.left_index,left_index)
            self.assertEqual(e.left_dimension,left_dimension)
            self.assertEqual(e.right_tensor_number,right_tensor_number)
            self.assertEqual(e.right_index,right_index)
            self.assertEqual(e.right_dimension,right_dimension)
    # }}}
    @with_checker
    def test_bad_rank_specification_detected(self,number_of_tensors=irange(1,5)): # {{{
        tensor_ranks = [randint(1,5) for _ in range(number_of_tensors)]
        observed_tensor_ranks = tensor_ranks
        while observed_tensor_ranks == tensor_ranks:
            observed_tensor_ranks = [randint(1,5) for _ in range(number_of_tensors)]
        try:
            formDataContractor([],[[(tensor_number,index)] for tensor_number, tensor_rank in enumerate(observed_tensor_ranks) for index in range(tensor_rank)],tensor_ranks=tensor_ranks)
            self.fail("exception was not thrown")
        except ValueError:
            pass
    # }}}
    @with_checker(number_of_calls=10)
    def test_triangle(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
        e = irange(1,10),
        f = irange(1,10),
    ):
        A = NDArrayData.newRandom(a,e,b)
        B = NDArrayData.newRandom(c,b,f)
        C = NDArrayData.newRandom(d,a,c)
        joins = [
            Join(0,0,2,1),
            Join(0,2,1,1),
            Join(1,0,2,2),
        ]
        shuffle(joins)
        self.assertDataAlmostEqual(
            A.contractWith(B,[2],[1]).contractWith(C,[0,2],[1,2]),
            formDataContractor(
                joins,
                [
                    [(0,1)],
                    [(1,2)],
                    [(2,0)],
                ]
            )(A,B,C)
        )
    # }}}
    @with_checker(number_of_calls=10)
    def test_1(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
        e = irange(1,10),
        f = irange(1,10),
    ):
        A = NDArrayData.newRandom(a,b,c)
        B = NDArrayData.newRandom(d,e,c,f)
        self.assertDataAlmostEqual(
            A.contractWith(B,[2],[2]).join([0,2],[1,3],[4]),
            formDataContractor(
                [Join(0,2,1,2)],
                [
                    [(0,0),(1,0)],
                    [(0,1),(1,1)],
                    [(1,3)],
                ]
            )(A,B)
        )
    # }}}
    @with_checker(number_of_calls=10)
    def test_2(self, # {{{
        a = irange(1,5),
        b = irange(1,5),
        c = irange(1,5),
        d = irange(1,5),
        e = irange(1,5),
        f = irange(1,5),
        g = irange(1,5),
        h = irange(1,5),
        i = irange(1,5),
    ):
        A = NDArrayData.newRandom(a,b,c,d)
        B = NDArrayData.newRandom(e,f,d,h,i)
        self.assertDataAlmostEqual(
            A.contractWith(B,[3],[2]).join([0,3],[1,4],[2,5],[6]),
            formDataContractor(
                [Join(0,3,1,2)],
                [
                    [(0,0),(1,0)],
                    [(0,1),(1,1)],
                    [(0,2),(1,3)],
                    [(1,4)],
                ]
            )(A,B)
        )
    # }}}
# }}}

class TestNormalize(TestCase): # {{{
    @with_checker # test_correctness {{{
    def test_correctness(self,number_of_dimensions=irange(2,4),size=irange(2,5)):
        index_to_normalize = randint(0,number_of_dimensions-1)
        normalized_tensor = normalize(crand(*(size,)*number_of_dimensions),index_to_normalize)
        indices_to_sum_over = list(range(number_of_dimensions))
        del indices_to_sum_over[index_to_normalize]
        should_be_identity = tensordot(normalized_tensor.conj(),normalized_tensor,(indices_to_sum_over,)*2)
        self.assertTrue(allclose(identity(size),should_be_identity))
    # }}}
    @with_checker # test_correctness_on_random_shape {{{
    def test_correctness_on_random_shape(self,number_of_dimensions=irange(2,6)):
        index_to_normalize = randint(0,number_of_dimensions-1)

        shape = [randint(1,5) for _ in range(number_of_dimensions)]
        try:
            normalized_tensor = normalize(crand(*shape),index_to_normalize)
        except ValueError:
            self.assertTrue(prod(shape[:index_to_normalize])*prod(shape[index_to_normalize+1:]) < shape[index_to_normalize])
            return

        indices_to_sum_over = list(range(number_of_dimensions))
        del indices_to_sum_over[index_to_normalize]
        should_be_identity = tensordot(normalized_tensor.conj(),normalized_tensor,(indices_to_sum_over,)*2)
        self.assertTrue(allclose(identity(normalized_tensor.shape[index_to_normalize]),should_be_identity))
    # }}}
    @with_checker # test_method_agreement {{{
    def test_method_agreement(self,number_of_dimensions=irange(2,4),size=irange(2,5)):
        index_to_normalize = randint(0,number_of_dimensions-1)
        tensor = crand(*(size,)*number_of_dimensions)
        normalized_tensor_1 = normalize(tensor,index_to_normalize)
        normalized_tensor_2 = multiplyTensorByMatrixAtIndex(tensor,computeNormalizerAndInverse(tensor,index_to_normalize)[0],index_to_normalize)
        self.assertTrue(allclose(normalized_tensor_1,normalized_tensor_2))
    # }}}
# }}}

class TestComputeNormalizerAndInverse(TestCase): # {{{
    @with_checker # test_inverse {{{
    def test_inverse(self,number_of_dimensions=irange(2,4),size=irange(2,5)):
        index_to_normalize = randint(0,number_of_dimensions-1)
        tensor = crand(*(size,)*number_of_dimensions)
        normalizer, inverse_normalizer = computeNormalizerAndInverse(tensor,index_to_normalize)
        self.assertTrue(allclose(identity(size),dot(normalizer,inverse_normalizer)))
    # }}}
    @with_checker # test_inverse_cancels_normalizer {{{
    def test_inverse_cancels_normalizer(self,number_of_dimensions=irange(2,4),size=irange(2,5)):
        index_to_normalize = randint(0,number_of_dimensions-1)
        tensor = crand(*(size,)*number_of_dimensions)
        normalizer, inverse_normalizer = computeNormalizerAndInverse(tensor,index_to_normalize)
        self.assertTrue(allclose(tensor,multiplyTensorByMatrixAtIndex(multiplyTensorByMatrixAtIndex(tensor,normalizer,index_to_normalize),inverse_normalizer,index_to_normalize)))
    # }}}
# }}}

class TestNormalizeAndReturnInverseNormalizer(TestCase): # {{{
    @with_checker
    def test_method_agreement(self,number_of_dimensions=irange(2,4),size=irange(2,5)):
        index_to_normalize = randint(0,number_of_dimensions-1)
        tensor = crand(*(size,)*number_of_dimensions)
        normalized_tensor_1 = normalize(tensor,index_to_normalize)
        normalizer, inverse_normalizer_1 = computeNormalizerAndInverse(tensor,index_to_normalize)
        normalized_tensor_2, inverse_normalizer_2 = normalizeAndReturnInverseNormalizer(tensor,index_to_normalize)
        self.assertTrue(allclose(normalized_tensor_1,normalized_tensor_2))
        self.assertTrue(allclose(inverse_normalizer_1,inverse_normalizer_1))
# }}}

class TestNormalizeAndDenormalize(TestCase): # {{{
    @with_checker
    def testEquivalence(self,number_of_dimensions=irange(2,4),size=irange(2,5)):
        index_to_normalize = randint(0,number_of_dimensions-1)
        index_to_denormalize = randint(0,number_of_dimensions-1)
        tensor_A, tensor_B = crand(2,*(size,)*number_of_dimensions)
        normalized_tensor_A_1, inverse_normalizer = normalizeAndReturnInverseNormalizer(tensor_A,index_to_normalize)
        unnormalized_tensor_B_1 = multiplyTensorByMatrixAtIndex(tensor_B,inverse_normalizer.transpose(),index_to_denormalize)
        normalized_tensor_A_2, unnormalized_tensor_B_2 = normalizeAndDenormalize(tensor_A,index_to_normalize,tensor_B,index_to_denormalize)
        self.assertTrue(allclose(normalized_tensor_A_1,normalized_tensor_A_2))
        self.assertTrue(allclose(unnormalized_tensor_B_1,unnormalized_tensor_B_2))
# }}}

class TestRelaxOver(TestCase): # {{{
    @with_checker # test_relaxOver_generalized {{{
    def test_relaxOver_generalized(self,N=irange(3,10)):
        matrix1 = NDArrayData.newRandomHermitian(N,N)
        matrix2 = NDArrayData.newRandomHermitian(N,N)
        matrix2 += -2*NDArrayData.newIdentity(N)*eigvalsh(matrix2.toArray())[0]
        old_vec = NDArrayData.newRandom(N).normalized()
        old_val = old_vec.conj().contractWith(matrix1.contractWith(old_vec,(1,),(0,)),(0,),(0,)).extractScalar().real/old_vec.conj().contractWith(matrix2.contractWith(old_vec,(1,),(0,)),(0,),(0,)).extractScalar().real
        new_vec = relaxOver(old_vec,Multiplier.fromMatrix(matrix1),Multiplier.fromMatrix(matrix2))
        new_val = new_vec.conj().contractWith(matrix1.contractWith(new_vec,(1,),(0,)),(0,),(0,)).extractScalar().real/new_vec.conj().contractWith(matrix2.contractWith(new_vec,(1,),(0,)),(0,),(0,)).extractScalar().real
        self.assertLess(new_val,old_val)
    # }}}
    @with_checker # test_relaxOver_standard {{{
    def test_relaxOver_standard(self,N=irange(3,10)):
        k = randint(1,N)
        matrix = NDArrayData.newRandomHermitian(N,N)
        old_vec = NDArrayData.newRandom(N).normalized()
        old_val = old_vec.conj().contractWith(matrix.contractWith(old_vec,(1,),(0,)),(0,),(0,)).extractScalar().real
        new_vec = relaxOver(old_vec,Multiplier.fromMatrix(matrix))
        new_val = new_vec.conj().contractWith(matrix.contractWith(new_vec,(1,),(0,)),(0,),(0,)).extractScalar().real
        self.assertLess(new_val,old_val)
    # }}}
# }}}
