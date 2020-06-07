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

# DONE - Working correctly.
def f(t, x, pk):
    """
    Describes heat sources through time applied at discrete points.
    
    Arguments:
        - t: time
        - x: position on the bar
        - pk: position pk in which 
          the force f_k(t,x) = r(t)ghk(x) will be applied
    """
    h = deltax
    r = 10*(1 + np.cos(5*t))
    #x = round(x, 7) # Why ?

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

# DONE - Working correctly.
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

# DONE - Working correctly.
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

    diagD, subdiagL = triDiagLDLtDecomposition(diagA, subdiagA)    # Now we can decompose A into 2 arrays: D and L
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

# DONE - Working correctly.
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

# DONE - Working correctly.
def buildNormalSystem(f, g):
    """
    Builds the normal system for the Least Squares Method. 
    Given a vector f, we would like to approximate it with vectors g1, g2, ..., gn.
    The solution to this problem is given by the normal system.
    Arguments:
    - f: desired vector
    - g: set of vectors used to approximate f
    Returns:
    - A: coefficient matrix of the normal system
    - b: independent vector of the normal system
    """

    n = g.shape[0]
    A = np.zeros((n,n))
    b = np.zeros(n)

    for i in range(n):
        for j in range(i, n):
            A[i, j] = np.dot(g[i], g[j])
            if i != j:
                A[j, i] = A[i, j]
        b[i] = np.dot(g[i], f)

    return A, b

def readTestFile(fileName):
    """
    Reads a .txt test file with the following format:
        <p1> <p2> <p3> ... <pnf>
        <uT[0]>
        <uT[1]>
        <uT[2]>
        ...
        <uT[N]>
    where pk represents the position where the force r(t)ghk(x) will be applied,
    and uT[k] represents a solution for the bar temperature problem at time t = T. 
    Arguments:
    - fileName: string that contains the name of the test file
    Returns:
    - p: array containing all positions pk.
    - uT: array containing the solution at t = T for all positions xk.
    """
    
    testFile = open(fileName, "r")
    uT = np.array([])
    firstLine = 1
    for line in testFile:
        if firstLine:
            p = line.split("       ") # What separates each pk in the test file given 
                                      # in the problem description
            p = np.array([float(pk) for pk in p])
            firstLine = 0
        else:
            uT = np.append(uT, float(line))
    testFile.close()

    return p, uT

def tempGraphs(u, onlyFinalResult=1):
    """
    Plots graphs related to the temperature: the evolution of the temperature in the bar through time, 
    and the temperature at t = T.
    Arguments:
        - u: 2-dimensional array that stores the temperature at each
          position xi and time tk
        - onlyFinalResult: determines if the graphs plotted are going
          to be only the last one, or the evolution as well.
    """
    M = u.shape[0] - 1
    N = u.shape[1] - 1
    deltax = 1/N

    step = int(M/10)

    if (onlyFinalResult == 0):
        fig = plt.figure()
        for i in range(0, M + 1, step):
            y = u[i, :]  
            x = np.linspace(0,N,N+1)*deltax
            plt.plot(x, y, label='t = ' + str(i/M))
        
        plt.legend()
        plt.suptitle('Evolução da temperatura u(t,x) com o tempo t')
        plt.xlabel('Comprimento da barra')
        plt.ylabel('Temperatura')
        evolucao = "evoluN=" + str(N) + "M=" + str(M) + ".png"
        fig.savefig(evolucao)
    
    fig = plt.figure()
    y = u[M, :]
    x = np.linspace(0,N,N+1)*deltax
    plt.plot(x, y, label='Temperatura com t = T')
    plt.xlabel('Comprimento da barra')
    plt.ylabel('Temperatura')
    plt.suptitle('Temperatura em t = T')
    fig.legend()
    final = "finalN=" + str(N) + "M=" + str(M) + ".png"
    fig.savefig(final)

    plt.show()

def printResults(p, a):
    """
    Prints the results of the problem: each position pk and its corresponding coefficient ak.
    """
    print("|  k  |    pk    |    ak    | ")
    print("|-----|----------|----------|")
    nf = p.shape[0]
    for k in range(nf):
        print("|{:^5d}|{:^10.3f}|{:^10.3f}|".format(k, p[k], a[k]))

# ---------------
# 1.2 Iterative Methods
# ---------------

# DONE - Working correctly.
def crankNicolson(u, T, pk):
    """
    The crank Nicolson method is described by equation (35) in the problem description.
    In a similar manner to the implicit Euler method, it calculates the evolution of u(t,x).
    However, this method has a second order convergence in both deltat and deltax.
    
    IMPORTANT: To improve readability, the function was updated to support only the initial 
    conditions specified in the problem description, that is, the initial conditions must be zero!

    Arguments:
        - u: 2-dimensional array that stores the temperature at each
          position xi and time tk
        - T: time interval
        - pk: point where the punctual force will be applied.
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
        b[0] = u[k, 1]*(1-lbd) + lbd/2*u[k, 2] + (deltat/2)*(f((k+1)*deltat, 1*deltax, pk) + f(k*deltat, 1*deltax, pk))
        for i in range (1,N-2):
            b[i] = u[k, i+1]*(1-lbd) + lbd/2*(u[k, i]+u[k, i+2]) + (deltat/2)*(f(k*deltat, (i+1)*deltax, pk)+f((k+1)*deltat,(i+1)*deltax, pk)) 
        b[N-2] = u[k, N-1]*(1-lbd) + lbd/2*(u[k, N-2]) + deltat/2*(f(k*deltat, (N-1)*deltax, pk) + f((k+1)*deltat, (N-1)*deltax, pk))
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

print()
print("Options: ")
print("    (a) Inverse problem verification # 1")
print("        | N = 128 | nf = 1 | p1 = 0.35 | a1 = 7 |")
print()

print("    (b) Inverse problem verification # 2")
print("        | N = 128 | nf = 4 | p1 = 0.15 | a1 = 2.3 |")
print("                           | p2 = 0.30 | a2 = 3.7 |")
print("                           | p3 = 0.70 | a3 = 0.3 |")
print("                           | p4 = 0.80 | a4 = 4.2 |")
print()
print("    (c) Use the existing solution given in 'teste.txt' without noise.")
print()
print()
print("    (d) Use the existing solution given in 'teste.txt' with noise.")
print()
option = input("Please input the letter corresponding to your choice: ")

if option == 'a' or option == 'b':
    if option == 'a':
        N = 128
        deltax = 1/N
        T = 1
        nf = 1
        p = np.array([0.35])
        a = np.array([7])
        
    elif option == 'b':
        N = 128
        deltax = 1/N
        T = 1
        nf = 4
        p = np.array([0.15, 0.30, 0.70, 0.80])
        a = np.array([2.3, 3.7, 0.3, 4.2])

    solutions = np.zeros((nf, N+1)) # We will store the solutions for each point in p here
    print()
    for k in range(nf):
        print("Calculating the solution for position p" + str(k+1) + ".")
        u0 = np.zeros((N+1, N+1))
        u = crankNicolson(u0, T, p[k])
        solutions[k] = u[N,:] # The solution at t = T
        solutions[k] = solutions[k][1:-1] # We must cut off the elements at the extremities!
        
    print("Calculating set of coefficients.")
    uT = sum(a[k]*solutions[k] for k in range(nf)) # Linear combination of the solutions
    A, b = buildNormalSystem(uT, solutions)
    a = solveLinearSystem(A, b)
    print()
    printResults(p, a)

elif option == 'c' or option == 'd':
    print()
    N = input("Please input the number of divisions in the bar length N: ")
    deltax = 1/N
    T = 1
    p, uTFile = readTestFile("teste.txt")
    nf = p.shape[0]
    uT = np.zeros((N+1,1))
    step = round((u.shape[0] - 1))/N
    i = 0
    for k in range(0, u.shape[0], step):
        uT[i] = uTFile[k]
        i = i + 1
    uT = uT[1:-1] # We must cut off the elements at the extremities!

    solutions = np.zeros((nf, N+1)) # We will store the solutions for each point in p here
    for k in range(nf):
        print("Calculating the solution for position p" + str(k+1) + ".")
        u0 = np.zeros((N+1, N+1))
        u = crankNicolson(u0, T, p[k])
        solutions[k] = u[N,:] # The solution at t = T
        solutions[k] = solutions[k][1:-1] # We must cut off the elements at the extremities!
    
    print("Calculating set of coefficients.")
    A, b = buildNormalSystem(uT, solutions)
    a = solveLinearSystem(A, b)
    print()
    printResults(p, a)

else: 
    print("Invalid option.")
