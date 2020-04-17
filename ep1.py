####################################################
# EP1 de MAP3121 - Metodos Numericos e Aplicacoes  #
#                                                  #
# Igor Augusto Gomes de Oliveira - 10773270        #
# Igor Nunes Ferro               - 10774138        #
####################################################

import numpy as np

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
    if type == 0: # Item (a)
        f = 10*x**2*(x - 1)- 60*x*t + 20*t # OK
    else if type == 1: # Item (b)
        f = 1 # item (b) - still undetermined
    else if type == 2: # Item (c)
        f = 2 # item (c) - still undetermined
    return f

def u0(x, type):
    """
    Describes the temperature at all positions x at t = 0.
    Arguments:
        - x : position on the bar
        - type: type of the function, specifies what u0(x) should be
          used at the function call
    """
    if type == 0: # Items (a) and (c)
        u0 = 0 # OK
    else if type == 1: # Item (b)
        u0 = 1 # item (b) - still undetermined
    return u0

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
    if type == 0: # Items (a) and (c)
        g1 = 0 # OK
    else if type == 1: # Items (b)
        g1 = 1 # item (b) - still undetermined
    return g1

def g2(t, type):
    """
    Describes the temperature at all times t at x = 1.
    Arguments:
        - t : time
        - type : type of the function, specifies what g2(t) should be
          used at the function call
    """
    if type == 0: # Items (a) and (c)
        g2 = 0 # OK
    else if type == 1: # Items (b)
        g2 = 1 # item (b) - still undetermined
    return g2

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

    #The most efficient way to generate que permutation matrix is using gauss elimination,
    #because with the multipliers we can refresh the matrix, therefore we can change the rows every time the multipliers are calculated

    n = len(A) #A.shape(0) didn't work when I used np.array
    P = np.eye(n) # Generate the identity matrix

    # It's necessary to apply the Gauss method on each row and column

    for i in range(0, n):
          greater = A[i][i]   # Defining the initial greater values
          greaterRow = i

          for j in range(i+1, n): # This "for" will find the greater value at column "i"
                                  # So it can be swapped for future use in the Gauss method, generating more accurate results

            if np.absolute(A[j][i]) > np.absolute(greater):
                                  # If the value at the row "j" is greater than the stored, it's going to refresh
                                  # the greater value, and in which row it is.
              greater = A[j][i]
              greaterRow = j

          # Now, it's necessary to swap the "i" row with the row that has the greater element in column "i"

          A[[i, greaterRow]] = A[[greaterRow, i]]
          P[[i, greaterRow]] = P[[greaterRow, i]]

          # And then, just apply the Gauss method for each row "j", starting at column "i + 1"

          for j in range(i+1, n):
            m = A[j][i] / A[i][i] # Calculating the multipliers
                                  # It's necessary to keep 2 rows in "memory" every iteraction: i and j
                                  # j is the row whose elements will be the minuends
                                  # i is the row whose elements will be the subtrahends
                                  # i will also be important as a column index

            for k in range(i, n): # "k" is a index for the columns and will scan each element of the row, starting at column "i"

              if i == k:
                A[j][k] = 0       # If the element is directly under the diagonal at column "i", it's going to be zero.

              elif i != k:
                A[j][k] -= A[i][k] * m
                                  # If the element is in any column, except "i", it's going to be subtracted by m * A[i][k]
              else:
                break # To avoid errors

    return(P)   # Since the purpose of this function is to generate the permutation matrix, we will only return it and not the upper triangular matrix A.

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

def method11(u, T, ftype=0):
    """
    Iterative method described by equation (11) in the problem description.
    It calculates the evolution of u(x,t) in time for all (xi,tk) interior
    elements.
    Arguments:
        - u : 2-dimensional array that stores the temperature at each
          position xi and time tk
        - T : time interval
        - ftype : f(x,t) function type
    """
    N = u.shape[0] - 1
    M = u.shape[1] - 1
    deltax = 1/N
    deltat = T/M
    for k in range(M):
        for i in range(1, N):
            u[i][k+1] = u[i][k] + deltat*((u[i-1][k] - 2*u[i][k] + u[i+1][k])/deltax**2 + f(i*deltax, k*deltat, ftype))
    return u

def implicitEuler(u, T, ftype=0, g1type=0, g2type=0):
    """
    The implicit Euler method is described by equation (29) in the problem description.
    It calculates the evolution of u(x,t), but the solution in a particular point of the
    grid depends on the solution on all other points. Thus, the linear system (29) must be solved.
    Arguments:
        - u : 2-dimensional array that stores the temperature at each
          position xi and time tk
        - T : time interval
        - ftype : f(x,t) function type
        - g1type : g1(t) function type
        - g2type : g2(t) function type
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
        b[0] = u[1][k] + deltat*f(1*deltax, (k+1)*deltat, ftype) + lbd*g1((k+1)*deltat, g1type)
        for l in range(1, N-2):
            b[l-1] = u[l][k] + deltat*f(l*deltax, (k+1)*deltat, ftype)
        b[N-2] = u[N-1][k] + deltat*f((N-1)*deltax, (k+1)*deltat, ftype) + lbd*g2((k+1)*deltat, g1type)

        # Solve Au = b, for time k+1
        u[1:N][k+1] = solveLinearSystem(A,b)

    return u

def crankNicholson(u, T, ftype=0, g1type=0, g2type=0):
    """
    The crank Nicholson method is described by equation (35) in the problem description.
    In a similar manner to the implicit Euler method, it calculates the evolution of u(x,t).
    However, this method has a second order convergence in both deltat and deltax.
    Arguments:
        - u : 2-dimensional array that stores the temperature at each
          position xi and time tk
        - T : time interval
        - ftype : f(x,t) function type
        - g1type : g1(t) function type
        - g2type : g2(t) function type
    """
    N = u.shape[0] - 1
    M = u.shape[1] - 1
    deltax = 1/N
    deltat = T/M
    lbd = deltat/deltax**2 # Lambda

    # Construct coefficient matrix A
    A = np.zeros((N-1,N-1))
    for i in range(N-1):
        A[i][i] = 1 + lbd
        if i != N - 2:
            A[i][i+1] = A[i+1][i] = -lbd/2

    for k in range(M):
        # Construct independent array b
        b = np.zeros((N-1,1))
        b[0] = (1 - lbd)*u[1][k] + lbd/2*(g1((k+1)*deltat, g1type) + g1(k*deltat, g1type) + u[2][k]) + deltat/2*(f(1*deltax, k*deltat, ftype) + f(1*deltax, (k+1)*deltat, ftype))
        for l in range(1, N-2):
            b[l-1] = u[l][k] + deltat*f(l*deltax, (k+1)*deltat, ftype)
        b[N-2] = (1 - lbd)*u[N-1][k] + lbd/2*(g2((k+1)*deltat, g2type) + g2(k*deltat, g2type) + u[N-2][k]) + deltat/2*(f((N-1)*deltax, k*deltat, ftype) + f((N-1)*deltax, (k+1)*deltat, ftype))

        # Solve Au = b, for time k+1
        u[1:N][k+1] = solveLinearSystem(A,b)

    return u

# =================================
# Simulations
# =================================

# ---------------
# Inputs
# ---------------

# To model the bar temperature problem, (x,t) will be understood as a (N,M) grid
# in which xi = i*deltax, i = 0, ..., N and tk = k*deltat, k = 0, ..., M, where
# deltax = 1/N and deltat = T/M.

print(" ______________________________ ")
print("|                              |")
print("|         MAP3121 - EP1        |")
print("|          Simulations         |")
print("|______________________________|")
print()
print("INPUTS")
print("---------------")

print()
input_list = input("Please input T, N and M respectively, separated by commas: ")
T, N, M = input_list.split(',')

deltax = 1/N
deltat = T/M
lbd = deltat/deltax**2 # Lambda

print()
print("Types :           '0'           |      '1'     |      '2'     ")
print("--------------------------------|--------------|--------------")
print("f(x,t): 10x^2(x-1) - 60xt + 20t | undetermined | undetermined ")
print("u0(x) :            0            | undetermined |     N.A.     ")
print("g1(t) :            0            | undetermined |     N.A.     ")
print("g2(t) :            0            | undetermined |     N.A.     ")
input_list = input("Please input f, g1 and g2 types respectively, separated by commas:")
ftype, u0type, g1type, g2type = input_list.split(',')

# ---------------
# Defining the grid
# ---------------

u = np.zeros((N+1, M+1))
# Applying initial conditions functions
u[:][0]
x = np.linspace(0, 1, N+1)
t = np.linspace(0, T, M+1)
