####################################################
# EP1 de MAP3121 - Metodos Numericos e Aplicacoes  #
#                                                  #
# Igor Augusto Gomes de Oliveira - 10773270        #
# Igor Nunes Ferro               - 10774138        #
####################################################

import numpy as np
import matplotlib.pyplot as plt

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
    func = 0

    if type == 0: # First function on item (a)
        func = 10*x**2*(x - 1)- 60*x*t + 20*t # OK
    elif type == 1: # Item (b)
        func = np.exp(t - x)*(25*t**2*cos(5*t*x))
    elif type == 2: # Item (c)
        h = 10000*(1 - 2*t**2)
        if np.abs(x - p) < h/2:
            func = 1/h
        else:
            func = 0
    elif type == 3: # Second function on item (a)
        func = 10*np.cos(10*t)*x**2*(1-x)**2-(1 + np.sin(10*t))*(12*x**2 - 12*x + 2)
      
    return func

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
    elif type == 1: # Item (b)
        u0 = np.exp(-x)
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
    elif type == 1: # Items (b)
        g1 = np.exp(t)
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
    elif type == 1: # Items (b)
        g2 = np.exp(t-1)*np.cos(5*t)
    return g2

def LDLtDecomposition(diagonalA, subdiagonalA):
    """
    Decomposes the matrix A into 2 matrices: L and D.
    The product L*D*L^t equals A.
    The matrix A is a symmetric tridiagonal matrix, so it can be described as 2 arrays: the diagonal and the subdiagonal.
    The matrix D is a diagonal matrix, it can be described as 1 array.
    The matrix L is a lower diagonal matrix with elements different from zero only in the diagonal and the subdiagonal
    Since the diagonal of the matrix L is composed only from ones, L can be described as 1 array, which is the subdiagonal.
    Arguments:
        - diagonalA: array that represents the diagonal of the matrix to be decomposed
        - subdiagonalA: array that represents the subdiagonal of the matrix to be decomposed
    Returns:
        - Larr: Array that represents the subdiagonal of the L matrix
        - Darr: Array that represents the diagonal of the D matrix
    """
    n = diagonalA.shape[0]   # First of all, we need to determine the size of the matrices, which is going to be the same as the matrix A

    A = np.eye(n)   # To use the algorithm, it's necessary to transform the arrays back to matrices.

    for i in range(n):
      A[i, i] = diagonalA[i]

    for i in range(n-1):
      A[i+1, i] = subdiagonalA[i+1]
      A[i, i+1] = subdiagonalA[i+1]

    # Now we have the original A matrix, which we can use for the decomposition.

    L = np.eye(n)   # We inicially generate an identity matrix as the L matrix.
                      # Since the L matrix is going to be a lower diagonal matrix, all the elements in its diagonal are 1.

    D = np.zeros((n,n)) # D is inicially adopted as a zero matrix, because it's a diagonal matrix, so only the elements
                        # that are in the diagonal can be different from zero.

    D[0, 0] = A[0, 0] # The first element of the diagonal from the D matrix is identical to the first diagonal element from A.

    # We can apply the Cholesky Decomposition to decompose a matrix "A" in two matrices "D" and "L", where A = L*D*Lt
    # The algorithm originally applies to a L*Lt decomposition, but there is an alternative form that generates a "D" matrix as well.

    for i in range(0, n): # At column 0, the elements will be "A" from the same position divided by "D[0 ,0]", which was previously determined.
      L[i, 0] = float(A[i, 0]) / float(D[0, 0])


    for i in range(1, n): # For the remaining rows, from 1 to n-1, we can apply the algorithm.
      for j in range(1, i+1):      # We need to apply it to every element, so it's necessary to apply to the columns from 1 to i (the diagonal).

        D[j, j] = A[j, j] - sum((L[j, k] ** 2) * D[k, k] for k in range(0, j))

        if i > j:
          L[i, j] = (1/D[j, j]) * (A[i, j] - sum(L[i, k]*L[j, k]*D[k, k] for k in range(0, j)))
                                  # Since there are no elements different from one in the diagonal at matrix L, the elements
                                  # of L will be only calculated with i > j.

    Darr = np.zeros(n)    # Now we can generate the arrays that are going to describe the D and L matrices.
    Larr = np.zeros(n)    # The size of Larr actually needs to be n-1, but we created it with size n because it's better for the loops.
                          # So the element at index 0 at Larrn is going to be zero, and won't be used in the future.

    for i in range(n):
      Darr[i] = D[i, i]

    for i in range(n-1):
      Larr[i+1] = L[i+1, i]

    return(Darr, Larr)

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

    n = A.shape[0]
    P = np.eye(n) # Generate the identity matrix

    # It's necessary to apply the Gauss method on each row and column

    for i in range(0, n):
          greater = A[i, i]   # Defining the initial greater values
          greaterRow = i

          for j in range(i+1, n): # This "for" will find the greater value at column "i"
                                  # So it can be swapped for future use in the Gauss method, generating more accurate results

            if np.absolute(A[j, i]) > np.absolute(greater):
                                  # If the value at the row "j" is greater than the stored, it's going to refresh
                                  # the greater value, and in which row it is.
              greater = A[j, i]
              greaterRow = j

          # Now, it's necessary to swap the "i" row with the row that has the greater element in column "i"

          A[[i, greaterRow]] = A[[greaterRow, i]]
          P[[i, greaterRow]] = P[[greaterRow, i]]

          # And then, just apply the Gauss method for each row "j", starting at column "i + 1"

          for j in range(i+1, n):
            m = A[j, i] / A[i, i] # Calculating the multipliers
                                  # It's necessary to keep 2 rows in "memory" every iteraction: i and j
                                  # j is the row whose elements will be the minuends
                                  # i is the row whose elements will be the subtrahends
                                  # i will also be important as a column index

            for k in range(i, n): # "k" is a index for the columns and will scan each element of the row, starting at column "i"

              if i == k:
                A[j, k] = 0       # If the element is directly under the diagonal at column "i", it's going to be zero.

              elif i != k:
                A[j, k] -= A[i, k] * m
                                  # If the element is in any column, except "i", it's going to be subtracted by m * A[i, k]
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

    n = A.shape[0]    # Since A is a symmetric tridiagonal, we can rearrange it into 2 arrays:
                      # The diagonal (diagA) and the subdiagonal (subdiagA)

    diagA = np.zeros(n)   # Creating both arrays with the size n
    subdiagA = np.zeros(n)  # subdiagA will actually be used starting at index 1

    for i in range(n):    # Setting the values on the arrays
      diagA[i] = A[i, i]

    for i in range(n-1):
      subdiagA[i+1] = A[i+1, i]

    # Get lower triangular matrix L and upper triangular matrix U

    #P = permutationMatrix(A)
    #A = np.dot(P, A)
    #A = np.dot(A, P.transpose())
    #b = np.dot(P,b)
    # Note: I don't think it's necessary to permutate the matrix, since A is symmetric and tridiagonal
    # The permutation only applies for the LU decomposition

    diagD, subdiagL = LDLtDecomposition(diagA, subdiagA)    # Now we can decompose A into 2 arrays: D and L
                                                            # diagD will represent the diagonal on a diagonal matrix, D
                                                            # subdiagL will represent the subdiagonal on a bidiagonal matrix L
                                                            # A can be described by the multiplication L * D * Lt

    L = np.eye(n)         # Now we can generate the matrices to transform back the arrays to matrices
    D = np.zeros((n,n))

    for i in range(n):
      D[i, i] = diagD[i]

    for i in range(n-1):
      L[i+1, i] = subdiagL[i+1]

    Lt = L.transpose()   # And we create the Lt matrix as well, which is the transposed L matrix

    # To find a solution for LDLt * x = b, we need to solve the system it by parts
    # First, we let y = Lt*x, and then we need to solve (L*D) * y = b

    LD = np.dot(L, D)

    y = np.zeros(n)

    for i in range(0, n):
        sumLD = 0
        for j in range(i):
            sumLD = sumLD + LD[i, j]*y[j]
        y[i] = (1/LD[i, i])*(b[i] - sumLD)

    # Now, we solve Lt*x = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sumLt = 0
        for j in range(n-1, i, -1):
            sumLt = sumLt + Lt[i, j]*x[j]
        x[i] = (1/Lt[i, i])*(y[i] - sumLt)

    return x

# ---------------
# Iterative Methods
# ---------------
# x <--> t CORRECTED
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
    M = u.shape[0] - 1
    N = u.shape[1] - 1
    deltat = T/M
    deltax = 1/N
    for k in range(M):
        for i in range(1, N):
            u[k+1, i] = u[k, i] + deltat*((u[k, i-1] - 2*u[k, i] + u[k, i+1])/deltax**2 + f(k*deltat, i*deltax, ftype))
    return u

# x <--> t CORRECTED
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
    M = u.shape[0] - 1
    N = u.shape[1] - 1
    deltat = T/M
    deltax = 1/N
    lbd = deltat/deltax**2 # Lambda

    # Construct coefficient matrix A
    A = np.zeros((N-1,N-1))
    for i in range(N-1):
        A[i, i] = 1 + 2*lbd
        if i != N - 2:
            A[i, i+1] = A[i+1, i] = -lbd

    for k in range(M):
        # Construct independent array b
        b = np.zeros((N-1,1))
        b[0] = u[k, 1] + deltat*f((k+1)*deltat, 1*deltax, ftype) + lbd*g1((k+1)*deltat, g1type)
        for l in range(1, N-2):
            b[l] = u[k, l] + deltat*f((k+1)*deltat, l*deltax, ftype)
        b[N-2] = u[k, N-1] + deltat*f((k+1)*deltat, (N-1)*deltax, ftype) + lbd*g2((k+1)*deltat, g1type)

        # Solve Au = b, for time k+1
        u[k+1, 1:N] = solveLinearSystem(A,b)

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
    M = u.shape[0] - 1
    N = u.shape[1] - 1
    deltat = T/M
    deltax = 1/N
    lbd = deltat/deltax**2 # Lambda

    # Construct coefficient matrix A
    A = np.zeros((N-1,N-1))
    for i in range(N-1):
        A[i, i] = 1 + lbd
        if i != N - 2:
            A[i, i+1] = A[i+1, i] = -lbd/2

    for k in range(M):
        # Construct independent array b
        b = np.zeros((N-1,1))
        b[0] = (1 - lbd)*u[1, k] + lbd/2*(g1((k+1)*deltat, g1type) + g1(k*deltat, g1type) + u[2, k]) + deltat/2*(f(1*deltax, k*deltat, ftype) + f(1*deltax, (k+1)*deltat, ftype))
        for l in range(1, N-2):
            b[l-1] = u[l, k] + deltat*f(l*deltax, (k+1)*deltat, ftype)
        b[N-2] = (1 - lbd)*u[N-1, k] + lbd/2*(g2((k+1)*deltat, g2type) + g2(k*deltat, g2type) + u[N-2, k]) + deltat/2*(f((N-1)*deltax, k*deltat, ftype) + f((N-1)*deltax, (k+1)*deltat, ftype))

        # Solve Au = b, for time k+1
        u[1:N, k+1] = solveLinearSystem(A,b)

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

T = int(T)
N = int(N)
M = int (M)

deltax = 1/N
deltat = T/M
lbd = deltat/deltax**2 # Lambda

print()
print("Types :           '0'           |      '1'     |      '2'     ")
print("--------------------------------|--------------|--------------")
print("f(t,x): 10x^2(x-1) - 60xt + 20t | undetermined | undetermined ")
print("u0(x) :            0            | undetermined |     N.A.     ")
print("g1(t) :            0            | undetermined |     N.A.     ")
print("g2(t) :            0            | undetermined |     N.A.     ")
input_list = input("Please input f, g1 and g2 types respectively, separated by commas:")
ftype, u0type, g1type, g2type = input_list.split(',')

ftype = int(ftype)
u0type = int(u0type)
g1type = int(g1type)
g2type = int(g2type)

# ---------------
# Defining the grid
# ---------------

u = np.zeros((M+1, N+1))
# Applying initial conditions functions
u[:][0]
x = np.linspace(0, 1, N+1)
t = np.linspace(0, T, M+1)

result = implicitEuler(u, T, ftype, g1type, g2type)

print(M)
fig = plt.figure()
plt.plot(result[M])
