import unittest
import numpy.testing as nptest

import itertools
import numpy as np

class Factor:
    # maintains the invariant that self.var is always in alphabetical
    # order, and contains no repeated entries

    def __init__(self, var, val):
        val = np.array(val)

        if len(var) != len(val.shape):
            raise ValueError("Mismatch between number of variables and dimensionality of value array")
        
        if len(var) > len(set(var)):
            raise ValueError("Repeated variable name")

        if len(var) == 0:
            self.var = var
            self.val = val
            self.cardinality = {}
        else:
            # put the variables in alphabetical order and get the
            # permutation required to do so, so that we can also adjust
            # the value array
            self.var, ordering = zip(
                *sorted(zip(var, range(len(var)))))

            self.var = var
            self.val = val.transpose(ordering)

            self.cardinality = dict(zip(self.var, self.val.shape))

    def __mul__(self, f):
        # Work out var order for result
        _var = sorted(list(set(self.var + f.var)))
        
        # Insert dimensions into self and f to be consistent with
        # shape of result
        _val1 = self.val.reshape(
            [self.cardinality.get(v,1) for v in _var])
        _val2 = f.val.reshape(
            [f.cardinality.get(v,1) for v in _var])

        # Multiply together and return result

        return Factor(var=_var, val=_val1*_val2)

    def __str__(self):
        _str = []
        for (inds, v) in \
                zip( itertools.product(*[range(self.cardinality[v]) 
                                         for v in self.var]), 
                     self.val.flatten()):
            line = ", ".join(["%s: %d"%p for p in zip(self.var, inds)])\
                +" -> "+str(v)
            _str.append(line)
        return "\n".join(_str)

    def toarray(self):
        return self.val

    def marginalise(self, reduce_over):
        reduce_over = set(reduce_over)
        _var = [x for x in self.var 
                if x not in reduce_over]
        if False:
            # we're marginalising over all values...
            return np.sum(self.val)
        else:
            sum_over = [i for(x, i) in zip(self.var, range(len(self.var)))
                        if x in reduce_over]
            new_shape = [self.cardinality[x] for x in self.var
                         if x not in reduce_over]
            _val = np.apply_over_axes(
                np.sum, self.val, sum_over)\
                .reshape(new_shape)
            return Factor(var=_var, val=_val)


class TestFactorObject(unittest.TestCase):
    def test_valid_create(self):
        f = Factor( var=['x','y'], 
                    val=np.array([[1.,2.],[3.,4.]]) )

    def test_create_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            Factor( var=['x','y'],
                    val=np.array([1.,2.]) )

    def test_create_repeated_vars(self):
        with self.assertRaises(ValueError):
            Factor( var=['x','y','x'],
                    val=np.arange(8).reshape(2,2,2) )

    def test_cardinality(self):
        f = Factor( var=['x','y','z'],
                    val=np.ones((2,3,4)) )
        self.assertEqual( f.cardinality, 
                          {'x':2, 'y':3, 'z':4} )

    def test_to_array_sorted(self):
        f = Factor( var=['x','y'],
                    val=np.arange(4).reshape(2,2) )
        nptest.assert_array_equal(
            f.toarray(), np.arange(4).reshape(2,2) )

    def test_to_array_unsorted(self):
        f = Factor( var=['y','x'],
                    val=np.arange(4).reshape(2,2) )
        nptest.assert_array_equal(
            f.toarray(), np.array([[0,2],[1,3]]) )

    def test_valid_product(self):
        f1 = Factor( var=['x'],
                     val=np.array([1,2]) )
        f2 = Factor( var=['x'],
                     val=np.array([3,4]) )
        f3 = Factor( var=['x','y'],
                     val=np.array([[1,2],[3,4]]) )
        f4 = Factor( var=['x','z'],
                     val=np.array([[1,2],[3,4]]) )
        self.assertIsInstance(f1*f2, Factor)
        nptest.assert_array_equal(
            (f1*f2).toarray(), np.array([3, 8]) )
        nptest.assert_array_equal(
            (f1*f3).toarray(), 
            np.array([[1,2],[6,8]]) )
        nptest.assert_array_equal(
            (f3*f4).toarray(),
            np.array([[[1,2],[2,4]],
                      [[9,12],[12,16]]])
            )

    def test_singleton_product(self):
        f1 = Factor( var=['x'],
                     val=np.array([1,2]) )
        fs = Factor( var=[],
                     val=3 )
        self.assertIsInstance(f1*fs, Factor)
        nptest.assert_array_equal(
            (f1*fs).toarray(),
            np.array([3,6]) )

        self.assertIsInstance(fs*f1, Factor)
        nptest.assert_array_equal(
            (fs*f1).toarray(),
            np.array([3,6]) )

    def test_invalid_products(self):
        f1 = Factor(var=['x'],val=np.array([1,2]))
        f2 = Factor(var=['x'],val=np.array([1,2,3]))
        with self.assertRaises(ValueError):
            f1*f2

    def test_marginalise(self):
        f1 = Factor(var=['x','y'], val=np.arange(4).reshape(2,2))
        self.assertIsInstance(f1.marginalise(['x']), Factor)
        nptest.assert_array_equal(
            f1.marginalise(['x']).toarray(),
            np.array([2,4]) )
        self.assertEqual(
            f1.marginalise(['x','y']).toarray(),
            6)

        f2 = Factor(var=['x','y'], val=np.arange(6).reshape(2,3))
        self.assertIsInstance(f2.marginalise(['x']), Factor)
        nptest.assert_array_equal(
            f2.marginalise(['x']).toarray(),
            np.array([3, 5, 7]) )
        nptest.assert_array_equal(
            f2.marginalise(['y']).toarray(),
            np.array([3, 12]) )

        f3 = Factor(var=['x','y','z'], val=np.arange(8).reshape(2,2,2))
        nptest.assert_array_equal(
            f3.marginalise(['x']).toarray(),
            np.array([[4, 6], [8, 10]]) )
        nptest.assert_array_equal(
            f3.marginalise(['x', 'y']).toarray(),
            np.array([12, 16]) )

if __name__=='__main__':
    unittest.main()
