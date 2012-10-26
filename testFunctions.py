#!/usr/bin/env python

import functions as F
import numpy as N
import unittest

class TestFunctions(unittest.TestCase):
    def testApproxJacobian1(self):
        slope = 3.0
        def f(x):
            return slope * x + 5.0
        x0 = 2.0
        dx = 1.e-3
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (1,1))
        self.assertAlmostEqual(Df_x, slope)

    def testApproxJacobian2(self):
        A = N.matrix("1. 2.; 3. 4.")
        def f(x):
            return A * x
        x0 = N.matrix("5; 6")
        dx = 1.e-6
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (2,2))
        N.testing.assert_array_almost_equal(Df_x, A)

    def testPolynomial(self):
        # p(x) = x^2 + 2x + 3
        p = F.Polynomial([1, 2, 3])
        for x in N.linspace(-2,2,11):
            self.assertEqual(p(x), x**2 + 2*x + 3)

    def testAprroxJacobian3(self):
        def f(x = N.matrix(N.zeros((2,1)))):
            ans = N.matrix(N.zeros((2,1)))
            ans[0,0] = x[0,0]**2 + 3.0*x[0,0]*x[1,0] + 4.0*x[1,0]**2
            ans[1,0] = -2.0*x[0,0]**2 + x[1,0]**2
            return ans
        x0 = N.matrix("1.; 2.")
        dx = 1.e-8
        Df_x = F.ApproximateJacobian(f, x0, dx)
        A = N.matrix("8. 19.; -4. 4.")
        N.testing.assert_array_almost_equal(Df_x, A)

    def testCompare(self):
        def f(x = N.matrix(N.zeros((2,1)))):
            ans = N.matrix(N.zeros((2,1)))
            ans[0,0] = x[0,0]**2 + 4.0*x[1,0]**2
            ans[1,0] = -2.0*x[0,0]**2 + x[1,0]**2
            return ans
        x0 = N.matrix("1.; 2.")
        dx = 1.e-8
       
        x = []
        x.append([N.matrix("1.0, 4.0"), N.matrix("2.0, 0.0; 0.0, 2.0")])
        x.append([N.matrix("-2.0, 1.0"), N.matrix("2.0, 0.0; 0.0, 2.0")])

        Df_x = F.AnalyticJacobian(x0, dx, ["Polynomial", x[0], x[1]])
        Df_x_A = F.ApproximateJacobian(f, x0, dx)

        N.testing.assert_array_almost_equal(Df_x, Df_x_A)

    def testAnalyticJacobian1(self):
        f = lambda x: x**2 - x + 2
        x0 = N.matrix('1.0')
        dx = 1.e-3
        Df_x = F.AnalyticJacobian(x0, dx, ["Polynomial", [N.matrix('1.0, -1.0'), N.matrix('2.0, -1')]])
        self.assertEqual(Df_x, 3.0)

    def testAnalyticJacobian2(self):
        f = lambda x: x**4 - x**2 + 2
        x0 = N.matrix('1.0')
        dx = 1.e-3
        Df_x = F.AnalyticJacobian(x0, dx, ["Polynomial", [N.matrix('1.0, -1.0'), N.matrix('4.0, 2.0')]])
        self.assertEqual(Df_x, 2.0)

    def testAnalyticJacobian3(self):
        def f(x = N.matrix(N.zeros((2,1)))):
            ans = N.matrix(N.zeros((2,1)))
            ans[0,0] = x[0,0]**2 + 4.0*x[1,0]**2
            ans[1,0] = -2.0*x[0,0]**2 + x[1,0]**2
            return ans
        x0 = N.matrix("1.; 2.")
        dx = 1.e-8
       
        x = []
        x.append([N.matrix("1.0, 4.0"), N.matrix("2.0, 0.0; 0.0, 2.0")])
        x.append([N.matrix("-2.0, 1.0"), N.matrix("2.0, 0.0; 0.0, 2.0")])

        Df_x = F.AnalyticJacobian(x0, dx, ["Polynomial", x[0], x[1]])
        exact = N.matrix("2. 16.; -4. 4.")
        N.testing.assert_array_almost_equal(Df_x, exact)

    def testAnalyticJacobian4(self):
        def f(x = N.matrix(N.zeros((2,1)))):
            ans = N.matrix(N.zeros((2,1)))
            ans[0,0] = x[0,0]**2 + 3.0*x[0,0]*x[1,0] + 4.0*x[1,0]**2
            ans[1,0] = -2.0*x[0,0]**2 - x[0,0]*x[1,0] + x[1,0]**2
            return ans
        x0 = N.matrix("1.; 2.")
        dx = 1.e-8

        x = []
        x.append([N.matrix("1.0, 3.0, 4.0"), N.matrix("2.0, 1.0, 0.0; 0.0, 1.0, 2.0")])
        x.append([N.matrix("-2.0, -1.0, 1.0"), N.matrix("2.0, 1.0, 0.0; 0.0, 1.0, 2.0")])

        Df_x = F.AnalyticJacobian(x0, dx, ["Polynomial", x[0], x[1]])
        exact = N.matrix("8. 19.; -6. 3.")
        N.testing.assert_array_almost_equal(Df_x, exact)

    def testAnalyticJacobian5(self):
        def f(x = N.matrix(N.zeros((2,1)))):
            ans = N.matrix(N.zeros((2,1)))
            ans[0,0] = M.pow(x[0,0], 0.5) + 2.0*M.pow(x[1,0], 0.5)
            ans[1,0] = 3.0*M.pow(x[0,0], 0.5) - M.pow(x[1,0], 0.5)
        x0 = N.matrix("1.0; 1.0")
        dx = 1.e-8

        x = []
        x.append([N.matrix("1.0, 2.0"), N.matrix("0.5, 0; 0, 0.5")])
        x.append([N.matrix("3.0, -1.0"), N.matrix("0.5, 0; 0, 0.5")])

        
        Df_x = F.AnalyticJacobian(x0, dx, ["Polynomial", x[0], x[1]])
        exact = N.matrix("0.5 1.; 1.5 -0.5")
        N.testing.assert_array_almost_equal(Df_x, exact)

    def testAnalyticJacobian6(self):
        def f(x = N.matrix(N.zeros((2,1)))):
            ans = N.matrix(N.zeros((2,1)))
            ans[0,0] = M.sin(x[0,0]) + 2.0*M.cos(x[1,0]) - M.pow(x[0,0], 2.0)
            ans[1,0] = 3.0*M.tan(x[0,0], 0.5) - M.tan(x[1,0])
        x0 = N.matrix("1.0; 1.0")
        dx = 1.e-12

        x = []
        x.append([{"sin": 1, 2: -1}, {"cos": 2}])
        x.append([{"tan": 3}, {"tan": -1}])

        Df_x = F.AnalyticJacobian(x0, dx, ["SimpleComplex", x[0], x[1]])
        exact = N.matrix("-0.45969769 -1.68294197; 7.27655646 -2.42551882")
        N.testing.assert_array_almost_equal(Df_x, exact)


if __name__ == '__main__':
    unittest.main()



