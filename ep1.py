####################################################
# EP1 de MAP3121 - Metodos Numericos e Aplicacoes  #
#                                                  #
# Igor Augusto Gomes de Oliveira - 10773270        #
# Igor Nunes Ferro               - 10774138        #
####################################################

import numpy as np

# =================================
# Inputs
# =================================

# ---------------
# Initial Parameters
# ---------------

# To model the bar temperature problem, (x,t) will be understood as a (N,M) grid
# in which xi = i*deltax, i = 0, ..., N and tk = k*deltat, k = 0, ..., M, where
# deltax = 1/N and deltat = T/M.

T = 1 # Time interval
N = 10 # Number of intervals in the lenght of the bar, 1
M = 10 # Number of intervals in the time period, T
deltax = 1/N
deltat = T/M
lbd = deltat/deltax**2 # Lambda

# ---------------
# Defining the grid
# ---------------

u = np.zeros((N+1, M+1))
x = np.linspace(0, 1, N+1)
t = np.linspace(0, T, M+1)

# =================================
# Functions and Iterative Methods
# =================================

# ---------------
# Functions
# ---------------

def f(t, x, type):
    """
    Describes heat sources through time.
    Arguments:
        - t : time
        - x : position on the bar
        - type : type of the function, specifies what f(x,t) should be
          used at the function call
    """

def u0(x, type):
    """
    Describes the temperature at all positions x at t = 0.
    Arguments:
        - x : position on the bar
        - type: type of the function, specifies what u0(x) should be
          used at the function call
    """

# g1(t) and g2(t) are Dirichlet boundary conditions, describing the
# temperature at the bar's extremities.
def g1(t, type):
    """
    Describes the temperature at all times t at x = 0.
    Arguments:
        - t : time
        - type : type of the function, specifies what g1(t) should be
          used at the function call
    """

def g2(t, type):
    """
    Describes the temperature at all times t at x = 1.
    Arguments:
        - t : time
        - type : type of the function, specifies what g2(t) should be
          used at the function call
    """

def LUDecomposition(A):
    """
    Performs LU decomposition.
    Arguments:
        - A : coefficient matrix
    Returns:
        - (L,U) : a tuple containing the lower triangular matrix and the upper
                  triangular matrix, respectively
    """
    n = A.shape[0]
    # Create zero matrices for L and U
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # Set all L[i][i] to 1
    for i in range(n):
            L[i][i] = 1

    # Select U[0][0] satisfying L[0][0]*U[0][0] = A[0][0]
    U[0][0] = A[0][0]

    # Set the first row of U and the first column of L
    for j in range(1, n):
        U[0][j] = A[0][j]
        L[j][0] = A[j][0]/U[0][0]

    # Set the remaining values (i.e the values in "the middle" of the matrices U and L)
    for j in range(1, n):

      for i in range(j+1):
          sumU = 0
          for k in range(i):
            sumU = sumU + (U[k][j] * L[i][k])

          U[i][j] = A[i][j] - sumU

      for i in range(j, n):
          sumL = 0
          for k in range(j):
            sumL = sumL + (U[k][j] * L[i][k])

          if i != j:    #The values in the diagonal of the L matrix are 1, so just checking to avoid errors if the last diagonal of U is 0
            L[i][j] = (A[i][j] - sumL) / U[j][j]

    return (L,U)

def permutationMatrix(A):
    """
    Returns the permutation matrix used in pivotal condensation.
    Arguments:
        - A: coefficient matrix
    Returns:
        - P: the permutation matrix
    """
    n = A.shape[0]

    # Create the initial permutation matrix, an identity matrix
    P = np.eye(n)

    

def solveLinearSystem(A,b):
    """
    Solves the linear system Ax = b.
    Arguments:
        - A : coefficient matrix
        - b : independent array
    Returns:
        - x : the solution to Ax = b.

    """
    n = A.shape[0]
    # Get lower triangular matrix L and upper triangular matrix U
    #P = permutationMatrix(A)
    #A = np.dot(P,A)
    #b = np.dot(P,b)
    L, U = LUDecomposition(A)

    # To solve LUx = b, we first must let y = Ux and solve Ly = b
    y = np.zeros(n)
    for i in range(0, n):
        sum = 0
        for j in range(i):
            sum = sum + L[i][j]*y[j]
        y[i] = (1/L[i][i])*(b[i] - sum)

    # Now, we solve Ux = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum = 0
        for j in range(n-1, i, -1):
            sum = sum + U[i][j]*x[j]
        x[i] = (1/U[i][i])*(y[i] - sum)

    return x
    
# ---------------
# Iterative Methods
# ---------------

def method11(u, T, f):
    """
    Iterative method described by equation (11) in the problem description.
    It calculates the evolution of u(x,t) in time for all (xi,tk) interior
    elements.
    Arguments:
        - u : 2-dimensional array that stores the temperature at each
          position xi and time tk
        - T : time interval
        - f : f(x,t) function
    """
    N = u.shape[0] - 1
    M = u.shape[1] - 1
    deltax = 1/N
    deltat = T/M
    for k in range(M):
        for i in range(1, N):
            u[i][k+1] = u[i][k] + deltat*((u[i-1][k] - 2*u[i][k] + u[i+1][k])/deltax**2 + f(i*deltax, k*deltat))

def implicitEuler(u, f, T, g1, g2):
    """
    The implicit Euler method is described by equation (29) in the problem description.
    It calculates the evolution of u(x,t), but the solution in a particular point of the
    grid depends on the solution on all other points. Thus, the linear system (29) must be solved.
    Arguments:
        - u : 2-dimensional array that stores the temperature at each
          position xi and time tk
        - T : time interval
        - f : f(x,t) function
        - g1 : g1(t) function
        - g2 : g2(t) function
    """
    N = u.shape[0] - 1
    M = u.shape[1] - 1
    deltax = 1/N
    deltat = T/M
    lbd = deltat/deltax**2 # Lambda

    # Construct coefficient matrix A
    A = np.zeros((N-1,N-1))
    for i in range(N-1):
        A[i][i] = 1 + 2*lbd
        if i != N - 2:
            A[i][i+1] = A[i+1][i] = -lbd

    for k in range(M):
        # Construct independent array b
        b = np.zeros((N-1,1))
        b[0] = u[1][k] + deltat*f(1*deltax, (k+1)*deltat) + lbd*g1((k+1)*deltat)
        for l in range(2, N-1):
            b[l-1] = u[l][k] + deltat*f(l*deltax, (k+1)*deltat)
        b[N-2] = u[N-1][k] + deltat*f((N-1)*deltax, (k+1)*deltat) + lbd*g2((k+1)*deltat)

    # TO-DO
