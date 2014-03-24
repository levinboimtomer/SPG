A python implementation of the spectral projected gradient (SPG) optimization method.

SPG is suited for optimizing differentiable real-valued multivariate functions
subject to simple constraints (namely, over a closed convex set)

The code is based on Mark Schmidt's minConf_SPG matlab implementation
http://www.di.ens.fr/~mschmidt/Software/minConf.html)

To test:
	python SPG_test.py

Incudes:
- Numerical differentiation
- Projection on a bounded range 
- Projection on the probability simplex (Duchi et al, "Efficient projections onto the l1-Ball for Learning in High Dimension") 

Require:
- cython (if you use the provided projections)

-- Tomer Levinboim
March 2014
