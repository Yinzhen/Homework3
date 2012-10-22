#!/usr/bin/env python

import newton
import unittest
import numpy as N
import math as M

class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
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
        

if __name__ == "__main__":
    unittest.main()
