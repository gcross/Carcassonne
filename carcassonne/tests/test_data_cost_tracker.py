# Imports {{{
from functools import reduce
from numpy import prod

from . import *
from ..data.cost_tracker import *
# }}}

class TestCostTracker(TestCase): # {{{
    @with_checker # test_outer_product {{{
    def test_outer_product(self,number_of_tensors=irange(1,5)):
        tensors = [CostTracker(randomShape(randint(1,3),3)) for _ in range(number_of_tensors)]
        contraction = formDataContractor([],[[(tensor_number,i)] for tensor_number, tensor in enumerate(tensors) for i in range(tensor.ndim)])(*tensors)

        correct_cost = 0
        current_size = prod(tensors[0].shape)
        for i in range(1,number_of_tensors):
            current_size *= prod(tensors[i].shape)
            correct_cost += current_size

        self.assertEqual(contraction.cost,correct_cost)
    # }}}
    @with_checker # test_two_tensors_inner_product_resulting_in_scalar {{{
    def test_two_tensors_inner_product_resulting_in_scalar(self,ndim=irange(1,5)):
        A_shape = randomShape(ndim)
        A = CostTracker(A_shape)
        B_permutation = randomPermutation(ndim)
        B_inverse_permutation = invertPermutation(B_permutation)
        B_shape = applyPermutation(B_permutation,A_shape)
        B = CostTracker(B_shape)
        axes = list(zip(range(ndim),B_inverse_permutation))
        shuffle(axes)
        A_axes, B_axes = zip(*axes)
        contraction = formDataContractor([Join(0,A_axes,1,B_axes)],[])(A,B)
        self.assertEqual(contraction.cost,prod(A_shape))
    # }}}
    @with_checker # test_two_tensors_inner_product_resulting_in_tensor {{{
    def test_two_tensors_inner_product_resulting_in_tensor(self,
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
        A = CostTracker(A_shape)

        B_ndim = middle_ndim + right_ndim
        B_permutation = randomPermutation(B_ndim)
        B_shape = applyPermutation(B_permutation,middle_shape + right_shape)
        B_axes = applyPermutation(invertPermutation(B_permutation),range(B_ndim))
        B_middle_axes, B_right_axes = B_axes[:middle_ndim], B_axes[middle_ndim:]
        B = CostTracker(B_shape)

        A_left_axes.sort()
        B_right_axes.sort()

        contraction = formDataContractor(
            [Join(0,A_middle_axes,1,B_middle_axes)],
            [[(tensor_number,axis)] for tensor_number, axes in enumerate([A_left_axes,B_right_axes]) for axis in axes]
        )(A,B)

        self.assertEqual(contraction.cost,prod(left_shape)*prod(middle_shape)*prod(right_shape))
    # }}}
    @with_checker # test_matrix_multiplication_two_matrices {{{
    def test_matrix_multiplication_two_matrices(self,m=irange(1,10),k=irange(1,10),n=irange(1,10)):
        A = CostTracker((m,k))
        B = CostTracker((k,n))
        contraction = formDataContractor([Join(0,1,1,0)],[[(0,0)],[(1,1)]])(A,B)
        self.assertEqual(contraction.cost,m*k*n)
    # }}}
    @with_checker # test_matrix_multiplication_four_matrices {{{
    def test_matrix_multiplication_four_matrices(self,
        a=irange(1,10),
        b=irange(1,10),
        c=irange(1,10),
        d=irange(1,10),
        e=irange(1,10),
    ):

        A = CostTracker((a,b))
        B = CostTracker((b,c))
        C = CostTracker((c,d))
        D = CostTracker((d,e))

        contraction = formDataContractor(
            [
                Join(0,1,1,0),
                Join(2,1,3,0),
                Join(1,1,2,0),
            ],
            [[(0,0)],[(3,1)]]
        )(A,B,C,D)
        self.assertEqual(contraction.cost,a*b*c+c*d*e+a*c*e)
    # }}}
    @with_checker # test_matrix_multiplication_three_matrices {{{
    def test_matrix_multiplication_three_matrices(self,
        a=irange(1,10),
        b=irange(1,10),
        c=irange(1,10),
        d=irange(1,10),
    ):
        A = CostTracker((a,b))
        B = CostTracker((b,c))
        C = CostTracker((c,d))
        joins = [
            Join(0,1,1,0),
            Join(1,1,2,0),
        ]
        contraction = formDataContractor(joins,[[(0,0)],[(2,1)]])(A,B,C)
        self.assertEqual(contraction.cost,a*b*c+a*c*d)
    # }}}
    @with_checker # test_matrix_multiplication_many_matrices {{{
    def test_matrix_multiplication_many_matrices(self, number_of_matrices=irange(1,10)):
        dimensions = [randint(1,10) for _ in range(number_of_matrices+1)]
        matrices = [CostTracker((dimensions[i],dimensions[i+1])) for i in range(number_of_matrices)]
        joins = [Join(i,1,i+1,0) for i in range(number_of_matrices-1)]
        contraction = formDataContractor(joins,[[(0,0)],[(number_of_matrices-1,1)]])(*matrices)

        correct_cost = 0
        a = dimensions[0]
        b = dimensions[1]
        for i in range(1,number_of_matrices):
            c = dimensions[i+1]
            correct_cost += a*b*c
            b = c

        self.assertEqual(contraction.cost,correct_cost)
    # }}}
    @with_checker # test_triangle {{{
    def test_triangle(self,
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
        e = irange(1,10),
        f = irange(1,10),
    ):
        A = CostTracker((a,e,b))
        B = CostTracker((c,b,f))
        C = CostTracker((d,a,c))
        contraction = formDataContractor(
            [
                Join(0,0,2,1),
                Join(0,2,1,1),
                Join(1,0,2,2),
            ],
            [
                [(0,1)],
                [(1,2)],
                [(2,0)],
            ]
        )(A,B,C)
        self.assertEqual(contraction.cost,a*b*c*d*e+b*c*d*e*f)
    # }}}
    @with_checker # test_1 {{{
    def test_1(self,
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
        e = irange(1,10),
        f = irange(1,10),
    ):
        A = CostTracker((a,b,c))
        B = CostTracker((d,e,c,f))
        contraction = \
            formDataContractor(
                [Join(0,2,1,2)],
                [
                    [(0,0),(1,0)],
                    [(0,1),(1,1)],
                    [(1,3)],
                ]
            )(A,B)
        self.assertEqual(contraction.cost,a*b*c*d*e*f)
    # }}}
    @with_checker # test_2 {{{
    def test_2(self,
        a = irange(1,5),
        b = irange(1,5),
        c = irange(1,5),
        d = irange(1,5),
        e = irange(1,5),
        f = irange(1,5),
        g = irange(1,5),
        h = irange(1,5),
    ):
        A = CostTracker((a,b,c,d))
        B = CostTracker((e,f,d,g,h))
        contraction = \
            formDataContractor(
                [Join(0,3,1,2)],
                [
                    [(0,0),(1,0)],
                    [(0,1),(1,1)],
                    [(0,2),(1,3)],
                    [(1,4)],
                ]
            )(A,B)
        self.assertEqual(contraction.cost,a*b*c*d*e*f*g*h)
    # }}}
# }}}
