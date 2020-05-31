####################################################
# EP2 de MAP3121 - Metodos Numericos e Aplicacoes  #
# Docente: André Salles de Carvalho                #
# Turma: 3                                         #
# Igor Augusto Gomes de Oliveira - 10773270        #
# Igor Nunes Ferro               - 10774138        #
####################################################

import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar

# =================================
# 1 Functions and Iterative Methods
# =================================

# ---------------
# 1.1 Functions
# ---------------

def f(t, x, p, k):
    """
    Describes heat sources through time applied at discrete points.
    Arguments:
        - t: time
        - x: position on the bar
        - p: array that contains the positions pk in which 
          the force f_k(t,x) = r(t)ghk(x) will be applied 
        - k: indicates which element in p will be selected
    """
    pk = p[k] 
    h = deltax
    
    r = 10*(1 + np.cos(5*t))

    x = round(x, 7) # Why ?

    if np.abs(x - pk) < h/2:
        f = r/h
    else:
        f = 0

    return f

def triDiagLDLtDecomposition(diagonalA, subdiagonalA):
    """
    Decomposes the tridiagonal matrix A into 2 matrices: L and D.
    The product L*D*L^t equals A. 
    Arguments:
        - diagonalA: array that represents the diagonal of the matrix to be decomposed
        - subdiagonalA: array that represents the subdiagonal of the matrix to be decomposed
    Returns:
        - Larr: array that represents the subdiagonal of the L matrix
        - Darr: array that represents the diagonal of the D matrix
    """
    n = diagonalA.shape[0]   # First of all, we need to determine the size of the matrices, which is going to be the same as the size of matrix A

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
      for j in range(1, i+1): # We need to apply it to every element, so it's necessary to apply to the columns from 1 to i (the diagonal).

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

# DONE - Working.
def LDLtDecomposition(A):
    """
    Decomposes the matrix A into 2 matrices: L and D.
    The product L*D*L^t equals A.
    Arguments:
        - A
    Returns:
        - L
        - D
    """
    n = A.shape[0]
    L = np.eye(n)
    D = np.zeros((n, 1))
    for i in range(0, n):
        D[i] = A[i, i] - np.dot(L[i, 0:i]**2, D[0:i])
        for j in range(i+1, n):
            L[j, i] = (A[j, i] - np.dot(L[j, 0:i]*L[i, 0:i], D[0:i])) / D[i]
    D = np.eye(n) * D
    return (D, L)

def triDiagSolveLinearSystem(diagA, subdiagA, b):
    """
    Solves the linear system Ax = b, where A is a tridiagonal matrix.
    Arguments:
        - diagA: diagonal of the coefficient matrix
        - subdiagA: subdiagonal of the coefficient matrix
        - b: independent array
    Returns:
        - x: the solution to Ax = b.
    """

    n = diagA.shape[0]

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

    # To find a solution for LDLt * x = b, we need to solve the system by parts
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

# DONE - Working
def solveLinearSystem(A, b):
    """
    Solves the linear system Ax = b.
    Arguments:
        - A: coefficient matrix
        - b: independent array
    Returns:
        - x: the solution to Ax = b.
    """
    n = A.shape[0]
    D, L = LDLtDecomposition(A) 
    Lt = L.transpose()   
    
    # To find a solution for LDLt * x = b, we need to solve the system by parts
    # First, we let y = Lt*x, and then we need to solve (L*D) * y = b
    LD = np.dot(L, D)
    y = np.zeros(n)
    for i in range(n):
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
# 1.2 Iterative Methods
# ---------------

# CHANGE NEEDED
def crankNicolson(u, T, ftype=0, g1type=0, g2type=0):
    """
    The crank Nicolson method is described by equation (35) in the problem description.
    In a similar manner to the implicit Euler method, it calculates the evolution of u(t,x).
    However, this method has a second order convergence in both deltat and deltax.
    Arguments:
        - u: 2-dimensional array that stores the temperature at each
          position xi and time tk
        - T: time interval
        - ftype: f(t,x) function type
        - g1type: g1(t) function type
        - g2type: g2(t) function type
    """
    M = u.shape[0] - 1
    N = u.shape[1] - 1
    deltat = T/M
    deltax = 1/N
    lbd = deltat/deltax**2 # Lambda

    # Construct coefficient matrix A. A is symmetric and tridiagonal, so it can be represented as 2 arrays of size N-1
    diagA = np.zeros(N-1)
    subdiagA = np.zeros(N-1)

    for i in range(N-1):
        diagA[i] = 1 + lbd 
        if i != N - 2:
            subdiagA[i+1] = -lbd/2 # subdiagA will actually be used starting at index 1

    bar = Bar("Running crankNicolson()", max=M) # This sets up a progress bar
    b = np.zeros((N-1))
    for k in range (0, M):
        b[0] = u[k, 1]*(1-lbd) + lbd/2*(g1(k*deltat, g1type) + g1((k+1)*deltat, g1type) + u[k, 2]) + (deltat/2)*(f((k+1)*deltat, 1*deltax, ftype) + f(k*deltat, 1*deltax, ftype))
        for i in range (1,N-2):
            b[i] = u[k, i+1]*(1-lbd) + lbd/2*(u[k, i]+u[k, i+2]) + (deltat/2)*(f(k*deltat, (i+1)*deltax, ftype)+f((k+1)*deltat,(i+1)*deltax, ftype)) 
        b[N-2] = u[k, N-1]*(1-lbd) + lbd/2*(g2((k+1)*deltat, g2type) + g2(k*deltat, g2type)+ u[k, N-2]) + deltat/2*(f(k*deltat, (N-1)*deltax, ftype) + f((k+1)*deltat, (N-1)*deltax, ftype))
        u[k+1, 1:N] = triDiagSolveLinearSystem(diagA, subdiagA, b)

        bar.next()
    bar.finish()

    return u  

# =================================
# 2 Simulations
# =================================

# To model the bar temperature problem, (x,t) will be understood as a (N,M) grid
# in which xi = i*deltax, i = 0, ..., N and tk = k*deltat, k = 0, ..., M, where
# deltax = 1/N and deltat = T/M.

print(" ________________________________________")
print("|                                        |")
print("|                Welcome to              |")
print("|              MAP3121 - EP2             |")
print("|               Simulations!             |")
print("|________________________________________|")
print()

input_list = input("Please input T, N and M respectively, separated by commas (e.g. 1,10,200): ")
T, N, M = input_list.split(',')

T = int(T)
N = int(N)
M = int (M)

deltax = 1/N
deltat = T/M
lbd = deltat/deltax**2 # Lambda

"""
# Select functions
# a1 is the first function described at item (a). a2 is the second function described at item (a)
print()
print("Types :           '0'               |                          '1'                             ")
print("------------------------------------|----------------------------------------------------------")
print("f(t,x): 10x^2(x-1) - 60xt + 20t (a1)|     5e^(t - x)*(5t^2*cos(5tx) - sin(5tx)*(x + 2t))    (b)")
print("u0(x) :            0         (a1, c)|                         e^(-x)                        (b)")
print("g1(t) :            0     (a1, a2, c)|                         e^(t)                         (b)")
print("g2(t) :            0     (a1, a2, c)|                         e^(t-1)                       (b)")
print()
print("Types :             '2'             |                          '3'                             ")
print("------------------------------------|----------------------------------------------------------")
print("f(t,x):   source at p = 0.25 (c)    | 10cos(10t)x^2(1-x)^2-(1 + sin(10t))(12x^2 - 12x + 2) (a2)")
print("u0(x) :             N.A.            |                      x^2(1-x)^2                      (a2)")
print("g1(t) :             N.A.            |                         N.A.                             ")
print("g2(t) :             N.A.            |                         N.A.                             ")
print()
input_list = input("Please input f, u0, g1 and g2 types respectively,\nseparated by commas (e.g. 2,0,0,0 for item (c)):")
ftype, u0type, g1type, g2type = input_list.split(',')

ftype = int(ftype)
u0type = int(u0type)
g1type = int(g1type)
g2type = int(g2type)

if (ftype not in [0,1,2,3]) or (u0type not in [0,1,3]) or (g1type not in [0,1]) or (g2type not in [0,1]):
    print()
    print("Invalid input.")

# Defining the grid
u = np.zeros((M+1, N+1))

# Applying initial conditions functions
for i in range(N+1):
    u[0,i] = u0(i*deltax, u0type)
for k in range(M+1):
    u[k,0] = g1(k*deltat, g1type)
    u[k,N] = g2(k*deltat, g2type)

# Select method
print()
print("Method Number |       Method    ")
print("--------------|-----------------")
print("      0       | explicitFD()    ")
print("      1       | implicitEuler() ")
print("      2       | crankNicolson() ")
print()
method = input("Please input the number corresponding to the method you would like to use: ")
print()

# Run methods
method = int(method)
if method == 0:
    result = explicitFD(u, T, ftype)
elif method == 1:
    result = implicitEuler(u, T, ftype, g1type, g2type)
elif method == 2:
    result = crankNicolson(u, T, ftype, g1type, g2type)
else:
    print("Invalid input.")


# Plot graphs
print("Plotting graphs")
tempGraphs(result)

# Calculate error norms
if (ftype != 2):
    errorNorms = np.zeros((M+1,1))
    truncErrorNorms = np.zeros((M+1,1))
    bar = Bar("Calculating error norms", max=M+1)
    for k in range(M+1):
        errorNorms[k] = errorNorm(k, u, T, ftype)
        truncErrorNorms[k] = truncErrorNorm(k, u, T, ftype, method)
        bar.next()
    bar.finish()

    resErrorNorm = errorNorms[M, 0] # Error norm "result". We want the error norm at t = T
    resTruncErrorNorm = np.amax(truncErrorNorms) # Truncation error norm "result". We want the maximum truncation error at all times
    
    print("Absolute error norm at t = T       : ", resErrorNorm)
    print("Truncation error norm at all times : ", resTruncErrorNorm)

    myfile = open("normasDosErros.txt", 'a')
    errorString = "Norma do erro absoluto em t = T                 -  N = " + str(N) + " e M = " + str(M) + ": " + str(resErrorNorm) + "\n"
    truncErrorString = "Norma do erro de truncamento em todos os tempos -  N = " + str(N) + " e M = " + str(M) + ": " + str(resTruncErrorNorm) + "\n\n"

    myfile.write(errorString)
    myfile.write(truncErrorString)

    myfile.close()

print()
"""
A = np.array([[4, -1, 1], [-1, 4.25, 2.75], [1, 2.75, 3.5]])
print(A)
b = np.array([1, 2, 3])
print(b)
x = solveLinearSystem(A, b)
print(x)