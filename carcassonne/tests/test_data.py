# Imports {{{
from . import *
from ..data import *
# }}}

class TestNDArrayData(TestCase): # {{{
    @with_checker # text_newEnlargener {{{
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
    @with_checker # test_increaseDimensions {{{
    def test_increaseDimensions(self,
        shape1 = (irange(1,5),)*5,
        shape2 = (irange(1,5),)*5,
        axis1 = irange(0,4),
        axis2 = irange(0,4),
        increment = irange(0,10),
    ):
        shape1 = list(shape1)
        shape2 = list(shape2)
        shape1[axis1] = shape2[axis2]
        data1 = NDArrayData.newRandom(*shape1)
        data2 = NDArrayData.newRandom(*shape2)
        data1_embiggened = data1.increaseDimensionAndFillWithRandom(axis1,shape1[axis1]+increment)
        data2_embiggened = data2.increaseDimensionsAndFillWithZeros((axis2,shape2[axis2]+increment))
        for i in range(5):
            if i == axis1:
                self.assertEqual(data1_embiggened.shape[i],data1.shape[i]+increment)
            else:
                self.assertEqual(data1_embiggened.shape[i],data1.shape[i])
            if i == axis2:
                self.assertEqual(data2_embiggened.shape[i],data2.shape[i]+increment)
            else:
                self.assertEqual(data2_embiggened.shape[i],data2.shape[i])
        self.assertDataAlmostEqual(
            data1.contractWith(data2,(axis1,),(axis2,)),
            data1_embiggened.contractWith(data2_embiggened,(axis1,),(axis2,)),
        )
    # }}}
    @with_checker(number_of_calls=9999) # test_normalizeAxis {{{
    def test_normalizeAxis(self,
        shape = (irange(1,5),)*5,
        axis = irange(0,4),
    ):
        print()
        print()
        data = NDArrayData.newRandom(*shape)
        received = data.normalizeAxis(axis,verbose=shape[axis]==1)
        data_normalized, normalizer, denormalizer = received
        #data_normalized, normalizer, denormalizer = data.normalizeAxis(axis,verbose=True)
        print("RcvReturning =",received[1:])
        print("N =",normalizer,"D =",denormalizer)
        print("Rcv1 =",received[1],"D =",received[2])
        _, X, Y = received
        print("X =",X,"Y =",Y)
        print("Rcv1 =",received[1],"D =",received[2])
        print("A: [1] =",data.normalizeAxis(axis,verbose=False)[1])
        self.assertDataAlmostEqual(
            data_normalized.absorbMatrixAt(axis,denormalizer.join(1,0)),
            data
        )
        print("B: [1] =",data.normalizeAxis(axis,verbose=False)[1])
        indices = list(range(5))
        del indices[axis]
        self.assertDataAlmostEqual(
            data_normalized.contractWith(data_normalized.conj(),indices,indices),
            NDArrayData.newIdentity(data_normalized.shape[axis])
        )
        print("C: [1] =",data.normalizeAxis(axis,verbose=False)[1])
        try:
            self.assertDataAlmostEqual(
                denormalizer.contractWith(normalizer,(1,),(1,)),
                NDArrayData.newIdentity(normalizer.shape[0])
            )
        except:
            print("D: [1] =",data.normalizeAxis(axis,verbose=False)[1])
            print(shape, axis)
            data.normalizeAxis(axis,verbose=True)
            print("N =",normalizer,"D =",denormalizer)
            print("[1] =",data.normalizeAxis(axis,verbose=False)[1])
            print(denormalizer.contractWith(normalizer,(1,),(1,)))
            raise
    # }}}
# }}}
