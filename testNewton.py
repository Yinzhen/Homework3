#!/usr/bin/env python

import newton
import unittest
import numpy as N
import math as M

class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=20)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)

    def testPolynomial(self):
    	f = lambda x: x**2 - x - 6
    	solver = newton.Newton(f, tol = 1.e-15, maxiter = 20)
    	x0 = [0, 2]
    	x_exact = [-2, 3]
    	for i in range(2):
    		x = solver.solve(x0[i])
    		self.assertEqual(x, x_exact[i])

    def testSysLinear(self):
    	A = N.matrix("2. 3.; 2. -1.")
    	B = N.matrix("-2; -3")
        def f(x):
            return A * x + B
        solver = newton.Newton(f, tol = 1.e-15, maxiter = 5)
        x0 = N.matrix("0; 0")
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, N.matrix("1.375; -.25"))

    def testParticular1(self):
        f = lambda x: M.sin(x) * M.sin(x) - .25
        solver = newton.Newton(f, tol = 1.e-15, maxiter = 30)
        x = solver.solve(1.)
        self.assertEqual(x, M.pi/6.)

    def testUsedAnalytical(self):
        f = lambda x: x**2 - x - 6.0
        solver = newton.Newton(f, tol = 1, maxiter = 1, dx = 1.e-6, Df = None)
        x = solver.solve(0.0)

    def testAnalyticSolution1(self):
        f = lambda x: x**2 - x - 6
        x0 = N.matrix('1.0')
        solver = newton.Newton(f, tol = 1.e-9, maxiter = 50, dx = 1.e-3, Df = ["Polynomial", [N.matrix('1.0, -1.0'), N.matrix('2.0, -1')]])
        x = solver.solve(x0)
        self.assertEqual(x, 3.0)

    def testAnalyticSolution2(self):
        def f(x = N.matrix(N.zeros((2,1)))):
            ans = N.matrix(N.zeros((2,1)))
            ans[0,0] = 10.0*x[0,0]**2 - x[1,0]**2 - x[0,0] + 1.0
            ans[1,0] = 2.0*x[0,0] - x[1,0] - 1.0
            return ans

        df = []
        df.append([N.matrix("10.0, -1.0, -1.0"), N.matrix("2.0, 0, 1.0; 0.0, 2.0, 0.0")])
        df.append([N.matrix("2.0, -1.0"), N.matrix("1.0, 0.0; 0.0, 1.0")])

        solver = newton.Newton(f, tol = 1.e-9, maxiter = 40, dx = 1.e-6, Df = ["Polynomial", df[0], df[1]])
        x0 = N.matrix("2.; -3.")
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, N.matrix("0.; -1."))
        

if __name__ == "__main__":
    unittest.main()
