import numpy as np
from ml import combine_mean, combine_variance
import math

class MLTest:
    def __init__(self):
        self.num_tests = 0

    def run(self):
        self.test_combine_mean()
        self.test_combine_variance()

    def test_combine_mean(self):
        # Test 1
        means_test = np.array(
            [
                [0.5,   1,      1.5,    2],
                [0.4,   0.9,    1.6,    2.2],
                [0.2,   0.1,    0.7,    0.6],
                [-1,    0,      3,      4]
            ]
        )
        means_result = np.array(
            [
                0.1/4,  2/4,    6.8/4,  8.8/4
            ]
        )
        assert np.all(list(map(math.isclose, combine_mean(means_test), means_result)))
        self.num_tests += 1

        # Test 2
        means_test = np.array(
            [
                [0.5,   1,      1.5,    2],
                [0.4,   0.9,    1.6,    2.2],
                [0.2,   0.1,    0.7,    0.6]
            ]
        )
        means_result = np.array(
            [
                (0.5+0.4+0.2)/3, (1+0.9+0.1)/3, (1.5+1.6+0.7)/3, (2+2.2+0.6)/3
            ]
        )
        assert np.all(list(map(math.isclose, combine_mean(means_test), means_result)))
        self.num_tests += 1

        # Test 3
        means_test = np.array(
            [
                [0.5,   1,      1.5,    2],
                [0.4,   0.9,    1.6,    2.2],
                [0.2,   0.1,    0.7,    0.6],
                [-1.2,  0.3,    0.1,    0.3],
                [3.2,   -2,     0.2,    0.6],
                [-2.2,  0.02,   2.3,    0]
            ]
        )
        means_result = np.array(
            [
                (0.5+0.4+0.2-1.2+3.2-2.2)/6,
                (1+0.9+0.1+0.3-2+0.02)/6,
                (1.5+1.6+0.7+0.1+0.2+2.3)/6,
                (2+2.2+0.6+0.3+0.6+0)/6
            ]
        )
        assert np.all(list(map(math.isclose, combine_mean(means_test), means_result)))
        self.num_tests += 1

         # Test 4
        means_test = np.array(
            [
                [0.5,   1,      1.5,    2,      1],
                [0.4,   0.9,    1.6,    2.2,    0.2],
                [0.2,   0.1,    0.7,    0.6,    0.3]
            ]
        )
        means_result = np.array(
            [
                (0.5+0.4+0.2)/3,
                (1+0.9+0.1)/3,
                (1.5+1.6+0.7)/3,
                (2+2.2+0.6)/3,
                (1+.2+.3)/3
            ]
        )
        assert np.all(list(map(math.isclose, combine_mean(means_test), means_result)))
        self.num_tests += 1

        # Test 5
        means_test = np.array(
            [
                [0.600118043,   0.5963963805,   0.5926362      ],
                [0.5219882356,  0.5257136305,   0.5294613641   ],
                [0.5875469977,  0.5863609844,   0.5851536621   ],
                [0.5903264195,  0.5886640837,   0.5869772172   ],
                [0.5897735053,  0.5851056066,   0.5808503213   ]
            ]
        )

        means_result = np.array(
            [
                (.600118043     + .5219882356 + .5875469977 + .5903264195 + .5897735053) / 5, 
                (.5963963805    + .5257136305 + .5863609844 + .5886640837 + .5851056066) / 5, 
                (.5926362       + .5294613641 + .5851536621 + .5869772172 + .5808503213) / 5, 
            ]
        )
        assert np.all(list(map(math.isclose, combine_mean(means_test), means_result)))
        self.num_tests += 1
        

    def test_combine_variance(self):
        # Test 1
        variances_test = np.array(
            [
                [0.5,   1,      1.5,    2,      4  ],
                [0.4,   0.9,    1.6,    2.2,    0.1],
                [0.2,   0.1,    0.7,    0.6,    1.2],
                [2,     0,      3,      4,      2.4]
            ]
        )
        means_test = np.array(
            [
                [0.2,   2,      2.5,    0,      2   ],
                [0.3,   1.4,    1.3,    1.2,    1   ],
                [0.1,   1.7,    1.5,    0.3,    4   ],
                [0.5,   3,      2,      1,      1.2 ]
            ]
        )

        combined_means_test = np.array(
            [
                (0.2 + 0.3 +  .1 + 0.5  ) /4,
                (2   + 1.4 + 1.7 + 3    ) /4,
                (2.5 + 1.3 + 1.5 + 2    ) /4,
                (0   + 1.2 + 0.3 + 1    ) /4,
                (2   + 1   + 4   + 1.2  ) /4
            ]
        )

        variances_result = np.array(
            [
                ((
                    (0.5    + (0.2  **2)) +
                    (0.4    + (0.3  **2)) +
                    (0.2    + (0.1  **2)) +
                    (2      + (0.5  **2))
                ) / 4) - (((0.2 + 0.3 +  .1 + 0.5)  /4) ** 2),
                
                ((
                    (1      + (2    **2)) +
                    (0.9    + (1.4  **2)) +
                    (0.1    + (1.7  **2)) +
                    (0      + (3    **2))
                ) / 4) - (((2   + 1.4 + 1.7 + 3)    /4) ** 2),

                ((
                    (1.5    + (2.5  **2)) +
                    (1.6    + (1.3  **2)) +
                    (0.7    + (1.5  **2)) +
                    (3      + (2    **2))
                ) / 4) - (((2.5 + 1.3 + 1.5 + 2)    /4) ** 2),

                ((
                    (2      + (0    **2)) +
                    (2.2    + (1.2  **2)) +
                    (0.6    + (0.3  **2)) +
                    (4      + (1    **2))
                ) / 4) - (((0   + 1.2 + 0.3 + 1)    /4) ** 2),

                ((
                    (4      + (2    **2)) +
                    (0.1    + (1    **2)) +
                    (1.2    + (4    **2)) +
                    (2.4    + (1.2  **2))
                ) / 4) - (((2   + 1   + 4   + 1.2)  /4) ** 2),
            ]
        )

        assert np.all(list(map(math.isclose, combine_variance(variances_test, means_test, combined_means_test), variances_result)))
        self.num_tests += 1

        # Test 2
        variances_test = np.array(
            [
                [4.06857259e-06, 2.07759347e-06, 1.31704905e-06],
                [1.32621600e-03, 1.17719888e-03, 1.03922509e-03],
                [1.90147046e-05, 1.10586714e-05, 5.94661498e-06],
                [2.07078401e-05, 1.23116389e-05, 6.78231680e-06],
                [1.65449096e-06, 1.31300582e-06, 1.36800076e-06]
            ]
        )
        means_test = np.array(
            [
                [0.60011804, 0.59639638, 0.59263623],
                [0.52198824, 0.52571363, 0.52946136],
                [0.587547,   0.58636098, 0.58515366],
                [0.59032642, 0.58866408, 0.58697722],
                [0.58977351, 0.58510561, 0.58085032]
            ]
        )

        combined_means_test = np.array(
            [
                0.57795064, 0.57644814, 0.57501576
            ]
        )

        variances_result = np.array(
            [
               ((
                   (4.06857259e-06 +    (0.60011804 **2)) + 
                   (1.32621600e-03 +    (0.52198824 **2)) + 
                   (1.90147046e-05 +    (0.587547   **2)) + 
                   (2.07078401e-05 +    (0.59032642 **2)) + 
                   (1.65449096e-06 +    (0.58977351 **2))
               ) / 5) - (0.57795064 **2),

               ((
                   (2.07759347e-06 +    (0.59639638 **2)) + 
                   (1.17719888e-03 +    (0.52571363 **2)) + 
                   (1.10586714e-05 +    (0.58636098 **2)) + 
                   (1.23116389e-05 +    (0.58866408 **2)) + 
                   (1.31300582e-06 +    (0.58510561 **2))
               ) / 5) - (0.57644814 **2),

               ((
                   (1.31704905e-06 +    (0.59263623 **2)) + 
                   (1.03922509e-03 +    (0.52946136 **2)) + 
                   (5.94661498e-06 +    (0.58515366 **2)) + 
                   (6.78231680e-06 +    (0.58697722 **2)) + 
                   (1.36800076e-06 +    (0.58085032 **2))
               ) / 5) - (0.57501576 **2),
            ]
        )
        assert np.all(list(map(math.isclose, combine_variance(variances_test, means_test, combined_means_test), variances_result)))
        self.num_tests += 1


if __name__ == "__main__":
    print("Running ml.py tests...")
    tests = MLTest()
    tests.run()
    print("All {} tests passed".format(tests.num_tests))