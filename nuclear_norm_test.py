import cvxpy as cp
import numpy as np

# Define the given matrix e
e = np.array([
    [0, 2, 3],
    [4, 0, 5],
    [6, 7, 0]
])

n = e.shape[0] # Get the size of the matrix

# Define the variable M
M = cp.Variable((n, n), symmetric=True)

# Define the objective
objective = cp.Minimize(cp.norm(M, 'nuc'))

# Define the constraints
constraints = [M[i, j] == e[i, j] for i in range(n) for j in range(n) if i != j]

# Define and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Print the optimal M
print(M.value)
