#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np

# Computes the minimum L2-distance projection of vector v onto the probability simplex
# REF: Efficient Projections onto the l1-Ball for Learning in High Dimensions. Duchi et al.
# def projectSimplex(np.ndarray[double, ndim=1]v):
#     cdef double sm, sm_row, theta
#     cdef int j, row
#     cdef np.ndarray[double, ndim=1] w, q
#     mu = np.sort(v)[::-1]
#     q = np.zeros(len(v))
#     sm = 0.0
#     for j in xrange(1, len(v)+1):
#         sm = sm+mu[j-1]
#         q[j-1] = mu[j-1] - (1.0/j)*(sm-1)
#         if q[j-1] > 0:
#             row = j
#             sm_row = sm
#     theta = (1.0/row)*(sm_row-1)
#     w = v-theta
#     w[w < 0] = 0
#     return w

def projectSimplex(v):
    mu = np.sort(v)[::-1]
    N = len(v)
    a_sm = mu.cumsum()
    q = mu - 1.0/np.array(range(1, N+1)) * (a_sm - 1)
    row = np.max(np.nonzero((q > 0))) + 1
    sm_row = a_sm[row-1]
    theta = (1.0/row)*(sm_row-1)
    w = v-theta
    w[w < 0] = 0
    return w


# Computes projection of x onto constraints LB <= x <= UB
def projectBound(x, LB, UB=None):
    x[x < LB] = LB[x < LB]
    if UB is not None:
        x[x > UB] = UB[x > UB]
    return x
