# Imports {{{
from scipy.linalg import eigh, eigvalsh
from scipy.sparse.linalg import eigs, eigsh

from . import *
from ..data import *
# }}}

class TestNDArrayData(TestCase): # {{{
    @with_checker # test_newEnlargener {{{
    def test_newEnlargener(self,old=irange(1,10),new=irange(10,20)):
        m1, m2 = NDArrayData.newEnlargener(old,new)
        self.assertEqual(m1.shape,(new,old))
        self.assertEqual(m2.shape,(new,old))
        self.assertDataAlmostEqual(m1.contractWith(m2,(0,),(0,)),m1.newIdentity(old))
        self.assertDataAlmostEqual(m2.contractWith(m1,(0,),(0,)),m2.newIdentity(old))
    # }}}
    @with_checker # test_absorbMatrixAt {{{
    def test_absorbMatrixAt(self,ndim=irange(1,5),n=irange(1,5)):
        axis = randint(0,ndim-1)
        tensor = NDArrayData.newRandom(*(randint(1,n) for _ in range(ndim)))
        matrix = NDArrayData.newRandom(tensor.shape[axis],tensor.shape[axis])
        new_axes = list(range(1,ndim))
        new_axes.insert(axis,0)
        self.assertDataAlmostEqual(
            tensor.absorbMatrixAt(axis,matrix),
            matrix.contractWith(tensor,(1,),(axis,)).join(*new_axes),
        )
    # }}}
    @with_checker # test_directSumWith_all_axes_summed {{{
    def test_directSumWith_all_axes_summed(self,shapes=((irange(1,5),)*5,)*2):
        datas = [NDArrayData.newRandom(*shape) for shape in shapes]
        scalars = [data.contractWith(data,range(5),range(5)).extractScalar() for data in datas]
        datasum = datas[0].directSumWith(datas[1])
        datasum_scalar = datasum.contractWith(datasum,range(5),range(5)).extractScalar()
        self.assertAlmostEqual(datasum_scalar,scalars[0]+scalars[1])
    # }}}
    @with_checker # test_directSumWith_some_axes_summed {{{
    def test_directSumWith_some_axes_summed(self,
        shapes=((irange(1,5),)*5,)*2,
        number_of_non_summed_axes=irange(0,4)
    ):
        non_summed_axes = randomSelectionFrom(number_of_non_summed_axes,range(5))
        summed_axes = [axis for axis in range(5) if axis not in non_summed_axes]

        shape0, shape1 = map(list,shapes)
        for axis in non_summed_axes:
            shape0[axis] = shape1[axis]
        shapes = (shape0,shape1)

        datas = [NDArrayData.newRandom(*shape) for shape in shapes]
        sums = [data.contractWith(data,summed_axes,summed_axes) for data in datas]
        total = sums[0]
        total += sums[1]

        datasum = datas[0].directSumWith(datas[1],*non_summed_axes)
        datasum_total = datasum.contractWith(datasum,summed_axes,summed_axes)

        self.assertDataAlmostEqual(datasum_total,total)
    # }}}
    @with_checker # test_increaseDimensions {{{
    def test_increaseDimensions_both(self,
        shapes = ((irange(1,5),)*5,)*2,
        number_of_axes = irange(0,5),
    ):
        axeses = [list(range(5)) for _ in range(2)]
        for axes in axeses:
            shuffle(axes)
        axeses = [axes[:number_of_axes] for axes in axeses]

        shapes = [list(shape) for shape in shapes]
        for axis0, axis1 in zip(*axeses):
            shapes[0][axis0] = shapes[1][axis1]
        increments = [randint(0,4) for _ in range(number_of_axes)]

        axes_and_new_dimensions_list = [
            [(axis,shape[axis]+increment) for axis, increment in zip(axes,increments)]
            for axes, shape in zip(axeses,shapes)
        ]
        data0, data1 = (NDArrayData.newRandom(*shape) for shape in shapes)
        data0_embiggened = data0.increaseDimensionsAndFillWithRandom(*axes_and_new_dimensions_list[0])
        data1_embiggened = data1.increaseDimensionsAndFillWithZeros(*axes_and_new_dimensions_list[1])
        datas = [data0,data1]
        data_embiggeneds = [data0_embiggened,data1_embiggened]
        for axes, data, data_embiggened in zip(axeses,datas,data_embiggeneds):
            for i in range(5):
                if i in axes:
                    self.assertEqual(data_embiggened.shape[i],data.shape[i]+increments[axes.index(i)])
                else:
                    self.assertEqual(data_embiggened.shape[i],data.shape[i])
        self.assertDataAlmostEqual(
            data0.contractWith(data1,axeses[0],axeses[1]),
            data0_embiggened.contractWith(data1_embiggened,axeses[0],axeses[1]),
        )
    # }}}
    @with_checker # test_normalizeAxis_with_sqrt_svals {{{
    def test_normalizeAxis_with_sqrt_svals(self,
        shape = (irange(1,5),)*5,
        axis = irange(0,4),
    ):
        data = NDArrayData.newRandom(*shape)
        normalizer, denormalizer = data.normalizeAxis(axis,True)
        self.assertDataAlmostEqual(
            denormalizer.contractWith(normalizer,(1,),(1,)),
            NDArrayData.newIdentity(normalizer.shape[0])
        )
    # }}}
    @with_checker # test_normalizeAxis_without_sqrt_svals {{{
    def test_normalizeAxis_without_sqrt_svals(self,
        shape = (irange(1,5),)*5,
        axis = irange(0,4),
    ):
        data = NDArrayData.newRandom(*shape)
        data_normalized, normalizer, denormalizer = data.normalizeAxis(axis)
        self.assertDataAlmostEqual(
            data_normalized.absorbMatrixAt(axis,denormalizer.join(1,0)),
            data
        )
        indices = list(range(5))
        del indices[axis]
        self.assertDataAlmostEqual(
            data_normalized.contractWith(data_normalized.conj(),indices,indices),
            NDArrayData.newIdentity(data_normalized.shape[axis])
        )
        self.assertDataAlmostEqual(
            denormalizer.contractWith(normalizer,(1,),(1,)),
            NDArrayData.newIdentity(normalizer.shape[0])
        )
    # }}}
    @with_checker # test_relaxOver_generalized {{{
    def test_relaxOver_generalized(self,N=irange(3,10)):
        k = randint(1,N)
        matrix1 = NDArrayData.newRandomHermitian(N,N)
        matrix2 = NDArrayData.newRandomHermitian(N,N)
        matrix2 += -2*NDArrayData.newIdentity(N)*eigvalsh(matrix2.toArray())[0]
        old_vec = NDArrayData.newRandom(N).normalized()
        old_val = old_vec.conj().contractWith(matrix1.contractWith(old_vec,(1,),(0,)),(0,),(0,)).extractScalar().real/old_vec.conj().contractWith(matrix2.contractWith(old_vec,(1,),(0,)),(0,),(0,)).extractScalar().real
        new_vec = old_vec.relaxOver(Multiplier.fromMatrix(matrix1),Multiplier.fromMatrix(matrix2),k=k)[0]
        new_val = new_vec.conj().contractWith(matrix1.contractWith(new_vec,(1,),(0,)),(0,),(0,)).extractScalar().real/new_vec.conj().contractWith(matrix2.contractWith(new_vec,(1,),(0,)),(0,),(0,)).extractScalar().real
        self.assertLess(new_val,old_val)
    # }}}
    @with_checker # test_relaxOver_standard {{{
    def test_relaxOver_standard(self,N=irange(3,10)):
        k = randint(1,N)
        matrix = NDArrayData.newRandomHermitian(N,N)
        old_vec = NDArrayData.newRandom(N).normalized()
        old_val = old_vec.conj().contractWith(matrix.contractWith(old_vec,(1,),(0,)),(0,),(0,)).extractScalar().real
        new_vec = old_vec.relaxOver(Multiplier.fromMatrix(matrix),k=k)[0]
        new_val = new_vec.conj().contractWith(matrix.contractWith(new_vec,(1,),(0,)),(0,),(0,)).extractScalar().real
        self.assertLess(new_val,old_val)
    # }}}
# }}}
