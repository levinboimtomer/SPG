from SPG import *
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
import cy_SPG

## TEST CODE
def squaredError(w, X, y):
    Xw = np.array(X.dot(w))
    res = Xw-y
    assert y.shape == res.shape


    f = (res ** 2).sum()
    g = 2*np.mat(X).T*res
    return f, g

def squaredError0(w, X, y):
    Xw = np.array(X.dot(w))
    res = Xw-y
    assert y.shape == res.shape

    f = (res ** 2).sum()
    return f

def my_test():
    test_mode = 0

    if test_mode == 1:  # TL: compared for correctness against the matlab implementation
        nVars = 3
        A = np.array([[0.5377, -1.3499, 0.6715],
             [1.8339,    3.0349, -1.2075],
             [-2.2588,    0.7254,    0.7172],
             [ 0.8622,   -0.0631,    1.6302],
             [ 0.3188,    0.7147,    0.4889],
             [-1.3077,   -0.2050,    1.0347],
             [-0.4336,   -0.1241,    0.7269],
             [ 0.3426,    1.4897,   -0.3034],
             [ 3.5784,    1.4090,    0.2939],
             [ 2.7694,    1.4172,   -0.7873]])
        x = np.array([0, 0, 0.1869])
        noise = 1.0933
  # Iteration   FunEvals Projections     Step Length    Function Val        Opt Cond
  #        1          3          4     1.00000e-03     1.31455e+01     1.27465e+01
  #        2          4          6     1.00000e+00     8.98746e+00     1.18163e+01
  #        3          5          8     1.00000e+00     6.53712e+00     7.42658e+00
  #        4          6         10     1.00000e+00     5.11521e+00     7.01375e+00
  #        5          7         12     1.00000e+00     4.29552e+00     5.09767e+00
  #        6          8         14     1.00000e+00     3.78106e+00     3.99576e+00
  #        7          9         16     1.00000e+00     3.51529e+00     2.37322e+00
  #        8         10         18     1.00000e+00     3.34049e+00     2.27406e+00
  #        9         11         20     1.00000e+00     3.23295e+00     2.33639e+00
  #       10         12         22     1.00000e+00     3.28995e+00     1.73562e+00
  #       11         13         24     1.00000e+00     3.13459e+00     7.19381e-01
  #       12         14         26     1.00000e+00     3.12497e+00     5.41564e-01
  #       13         15         28     1.00000e+00     3.11916e+00     4.25801e-01
  #       14         16         30     1.00000e+00     3.10916e+00     5.44766e-02
  #       15         17         32     1.00000e+00     3.11021e+00     3.51837e-01
  #       16         18         34     1.00000e+00     3.10913e+00     6.90368e-04
  #       17         19         36     1.00000e+00     3.10913e+00     1.19318e-05
    else:
        ## larger problem.
        nInst = 24
        nVars = 40
        A = np.random.randn(nInst,nVars);
        x = np.random.rand(nVars,) * (np.random.rand(nVars,) > .5)
        noise = np.random.randn(1,)

    # This test minimizes the squared error Ax-b, subject to x being non-negative.
    b = np.array(x*np.mat(A.T) + noise)
    x_init = np.zeros((nVars,))
    LB = np.zeros((nVars,))
    UB = np.ones((nVars,)) * np.inf
    print b.shape
    spg_options = default_options
    spg_options.curvilinear = 1
    spg_options.interp = 2
    spg_options.numdiff = 1 # 0 to use gradients, 1 for numerical diff

    if spg_options.numdiff == 0:
        funObj = lambda x: squaredError(x, A, b)
    else:
        funObj = lambda x: squaredError0(x, A, b)

    funProj = lambda x: cy_SPG.projectBound(x, LB, UB)

    x, f = SPG(funObj, funProj, x_init, spg_options)
    return x, f, A, b 

if __name__ == '__main__':
    import random
    np.random.seed(2)
    x, f, A, b = my_test()
