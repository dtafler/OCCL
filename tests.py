import unittest
import utilities

class TestLoss(unittest.TestCase):

    def setUp(self):
        print('setup TestLoss')
        
    def tearDown(self):
        print('tearDown TestLoss\n')

    def test_decode_kf_classification(self):
        placeCoding = ['red, square, square', 'red, circle, square', 'red, triangle, square', 'yellow, square, square', 'yellow, circle, square', 'yellow, triangle, square', 'blue, square, square', 'blue, circle, square', 'blue, triangle, square', 'red, square, circle', 'red, circle, circle', 'red, triangle, circle', 'yellow, square, circle', 'yellow, circle, circle', 'yellow, triangle, circle', 'blue, square, circle', 'blue, circle, circle', 'blue, triangle, circle', 'red, square, triangle', 'red, circle, triangle', 'red, triangle, triangle', 'yellow, square, triangle', 'yellow, circle, triangle', 'yellow, triangle, triangle', 'blue, square, triangle', 'blue, circle, triangle', 'blue, triangle, triangle', 'red, square, no_shape', 'red, circle, no_shape', 'red, triangle, no_shape', 'yellow, square, no_shape', 'yellow, circle, no_shape', 'yellow, triangle, no_shape', 'blue, square, no_shape', 'blue, circle, no_shape', 'blue, triangle, no_shape']
        for i in range(35):
            color, small, large = utilities.decode_kf_classification(i)
            trueValues = placeCoding[i].split(sep=', ')
            self.assertEqual(color, trueValues[0])
            self.assertEqual(small, trueValues[1])
            self.assertEqual(large, trueValues[2])
        
    def test_get_all_but_first_args(self):
        self.assertEqual(utilities.get_all_but_first_args("pred(arg1,arg2, arg3,arg4)."), ["arg2", "arg3", "arg4"])

    def test_concat_labels(self):
        self.assertEqual(utilities.concat_labels(['1', '2', '3', '4', '5', '6', '7']), ['1, 2, 3', '4', '5', '6', '7'])

    
    #TODO: tests for get_combos and get_filter_combos, get_distance
        
if __name__ == '__main__':
    unittest.main()