import numpy as np
import cvxpy as cp
# import cvxopt
def solver_fixedb(X, b, p):
    B = np.diag(b)
    Y = X @ B
    D = X.shape[0]
    q = cp.Variable(D) 
    cost = cp.sum(cp.power(Y.T @ q, p))
    constraints = [cp.norm(q) <= 1, Y.T @ q >= 0 ]
    prob = cp.Problem(cp.Maximize(cost), constraints)
    try:
        prob.solve(solver = 'CVXOPT' , verbose = False)
        q = q.value
        metric = prob.value
    except:
        q = np.zeros(D)
        metric = 0
    return q, metric 