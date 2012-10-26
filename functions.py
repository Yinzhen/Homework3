import numpy as N
import math as M

def ApproximateJacobian(f, x, dx=1e-6):
    """Return an approximation of the Jacobian Df(x) as a numpy matrix"""
    try:
        n = len(x)
    except TypeError:
        n = 1
    fx = f(x)
    Df_x = N.matrix(N.zeros((n,n)))
    for i in range(n):
        v = N.matrix(N.zeros((n,1)))
        v[i,0] = dx
        Df_x[:,i] = (f(x + v) - fx)/dx
    return Df_x

def AnalyticJacobian(x, dx=1e-6, function = []):
    try:
        n = len(x)
    except TypeError:
        n = 1
    Df_x = N.matrix(N.zeros((n,n)))

    def Df(x, function, i = 0): 
        if function[0] == "test":
            print "Having used analytical Jacobian"

        elif function[0] == "Polynomial":
            coeffs = function[i][0]
            n = coeffs.shape[1]
            dimen = len(x)
            ans = N.matrix(N.zeros((1, dimen)))
            order_coeff = N.matrix(N.zeros((n, 1)))
            for j in range(dimen):

                deriv = function[i][1].copy()
                order = function[i][1].copy()
                order[j, :] = order[j, :] - 1.0
                for k1 in range(n): 
                    temp = 1 
                    for k2 in range(dimen):
                        temp = temp*M.pow(x[k2], order[k2,k1])
                    ans[0, j] = ans[0, j] + coeffs[0, k1]*deriv[j, k1]*temp
            return ans

        elif function[0] == "SimpleComplex":
            dimen = len(x)
            ans = N.matrix(N.zeros((1, dimen)))
            for j in range(dimen):
                coeffs = function[i][j]
                for para in coeffs.keys():
                    df = BasicDerivative(para)
                    ans[0, j] = ans[0, j] + coeffs[para]*df(x[i-1])
            return ans

        else:
            print "There is no analytical Jacobian, please chage Df in test function as False"


    for i in range(n):
        Df_x[i,:] = Df(x, function, i+1)
    return Df_x

class Polynomial(object):
    """Callable polynomial object.

    Example usage: to construct the polynomial p(x) = x^2 + 2x + 3,
    and evaluate p(5):

    p = Polynomial([1, 2, 3])
    p(5)"""

    def __init__(self, coeffs):
        self._coeffs = coeffs

    def __repr__(self):
        return "Polynomial(%s)" % (", ".join([str(x) for x in self._coeffs]))

    def f(self,x):
        ans = self._coeffs[0]
        for c in self._coeffs[1:]:
            ans = x*ans + c
        return ans

    def __call__(self, x):
        return self.f(x)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class BasicDerivative(object):

    def __init__(self, function):
        self._function = function

    def __repr__(self):
        return "BasicDerivative(%s)" % (self._function)

    def f(self, x):
        if self._function == "sin":
            return M.cos(x)
        elif self._function == "cos":
            return -M.sin(x)
        elif self._function == "tan":
            return M.pow(1.0/M.tan(x), -2)
        elif is_number(self._function):
            return M.pow(x, float(self._function) - 1)
        else:
            raise Exception("No such derivative function")

    def __call__(self, x):
        return self.f(x)
