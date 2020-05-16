####################################################
# EP1 de MAP3121 - Metodos Numericos e Aplicacoes  #
#                                                  #
# Igor Augusto Gomes de Oliveira - 10773270        #
# Igor Nunes Ferro               - 10774138        #
####################################################

import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar

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
    p = 0.25
    h = deltax

    if type == 0: # First function on item (a)
        func = 10*x**2*(x - 1)- 60*x*t + 20*t 
    elif type == 1: # Item (b)
        func = 5*(np.exp(t - x))*(5*t**2*np.cos(5*t*x) - np.sin(5*t*x)*(x + 2*t))
    elif type == 2: # Item (c)
        r = 10000*(1 - 2*t**2)
        if np.abs(x - p) < h/2:
            func = r/h
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
    if type == 0 or type == 2: # Items (a) and (c)
        u0 = 0 
    elif type == 1: # Item (b)
        u0 = np.exp(-x)
    elif type == 3: # Second function of item (a)
        u0 = x**2 * (1-x)**2
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
    if type == 0 or type == 3 or type == 2: # Items (a) and (c)
        g1 = 0 
    elif type == 1: # Item (b)
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
    if type == 0 or type == 3 or type == 2: # Items (a) and (c)
        g2 = 0 
    elif type == 1: # Item (b)
        g2 = np.exp(t-1)*np.cos(5*t)
    return g2

def uExact(t, x, type):
    """
    Describes the exact temperature at time t and distance x.
    Arguments:
        - t : time
        - type : type of the function, specifies what uExact(t,x) should be
         used at the function call
    """
    if type == 0:  # First function of item (a)
        exact = 10*t * x**2 * (x-1)
    elif type == 1: # Item (b)
        exact = np.exp(t-x)*np.cos(5*t*x)
    elif type == 3: # Second function of item (a)
        exact = (1 + np.sin(10*t)) * x**2 * (1-x)**2

    return exact

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

def explicitFD(u, T, ftype=0):
    """
    Explicit finite differences method, described by equation (11) in the problem description.
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

    bar = Bar("Running explicitFD()", max=M)
    for k in range(M):
        for i in range(1, N):
            u[k+1, i] = u[k, i] + deltat*((u[k, i-1] - 2*u[k, i] + u[k, i+1])/deltax**2 + f(k*deltat, i*deltax, ftype))
        bar.next()
    bar.finish()

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

    bar = Bar("Running implicitEuler()", max=M)
    for k in range(M):
        # Construct independent array b
        b = np.zeros((N-1,1))
        b[0] = u[k, 1] + deltat*f((k+1)*deltat, 1*deltax, ftype) + lbd*g1((k+1)*deltat, g1type)
        for l in range(1, N-2):
            b[l] = u[k, l+1] + deltat*f((k+1)*deltat, (l+1)*deltax, ftype)
        b[N-2] = u[k, N-1] + deltat*f((k+1)*deltat, (N-1)*deltax, ftype) + lbd*g2((k+1)*deltat, g2type)

        # Solve Au = b, for time k+1
        u[k+1, 1:N] = solveLinearSystem(A,b)
        
        bar.next()
    bar.finish()

    return u

def crankNicolson(u, T, ftype=0, g1type=0, g2type=0):
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

    bar = Bar("Running crankNicolson()", max=M)
    b = np.zeros((N-1))
    
    for k in range (0, M):
        b[0] = u[k, 1]*(1-lbd) + lbd/2*(g1(k*deltat, g1type) + g1((k+1)*deltat, g1type) + u[k, 2]) + (deltat/2)*(f((k+1)*deltat, 1*deltax, ftype) + f(k*deltat, 1*deltax, ftype))
        for i in range (1,N-2):
            b[i] = u[k, i+1]*(1-lbd) + lbd/2*(u[k, i]+u[k, i+2]) + (deltat/2)*(f(k*deltat, (i+1)*deltax, ftype)+f((k+1)*deltat,(i+1)*deltax, ftype)) 
        b[N-2] = u[k, N-1]*(1-lbd) + lbd/2*(g2((k+1)*deltat, g2type) + g2(k*deltat, g2type)+ u[k, N-2]) + deltat/2*(f(k*deltat, (N-1)*deltax, ftype) + f((k+1)*deltat, (N-1)*deltax, ftype))
        u[k+1, 1:N] = solveLinearSystem(A,b)

        bar.next()
    bar.finish()

    return u

def tempGraphs(u):
    """
    Plots graphs related to the temperature: the evolution of the temperature in the bar through time, 
    and the temperature at t = T.
    Arguments:
        - u : 2-dimensional array that stores the temperature at each
          position xi and time tk
    """
    M = u.shape[0] - 1
    N = u.shape[1] - 1
    deltax = 1/N

    step = int(M/10)

    fig = plt.figure()
    for i in range(0, M + 1, step):
        y = u[i,...]  
        x = np.linspace(0,N,N+1)*deltax
        plt.plot(x, y, label='t = ' + str(i/M))
    
    plt.legend()
    plt.suptitle('Evolução da temperatura u(t,x) com o tempo t')
    plt.xlabel('Comprimento da barra')
    plt.ylabel('Temperatura')
    evolucao = "evoluN=" + str(N) + "M=" + str(M) + ".png"
    fig.savefig(evolucao)
    
    fig = plt.figure()
    y = u[M,...]
    x = np.linspace(0,N,N+1)*deltax
    plt.plot(x, y, label='Temperatura com t = T')
    plt.xlabel('Comprimento da barra')
    plt.ylabel('Temperatura')
    plt.suptitle('Temperatura em t = T')
    fig.legend()
    final = "finalN=" + str(N) + "M=" + str(M) + ".png"
    fig.savefig(final)

    plt.show()

def errorGraph(e):
    """
    Plots an error graph according to the error vector e
    Arguments:
        - e : error array, of size M
    """
    M = e.shape[0]
    k = np.arange(M)

    fig = plt.figure()
    plt.plot(k, e, label='Erro ')
    plt.legend()
    plt.suptitle('Evolução do erro com a iteração k')
    plt.xlabel('Número da iteração')
    plt.ylabel('Erro')
    #evolucao = "evoluN=" + str(N) + "M=" + str(M) + ".png"
    #fig.savefig(evolucao)
    plt.show()

def errorNorm(k, u, T, ftype):
    """
    Calculates the norm of the absolute error at the instant tk = k*deltaT.
    Arguments:
        - k
        - u : 2-dimensional array that stores the temperature at each
          position xi and time tk
        - T : time period
        - utype : function type, it will determine what is the exact function.
    """
    M = u.shape[0] - 1
    N = u.shape[1] - 1
    deltax = 1/N
    deltat = T/M

    exactSolution = np.zeros((N+1))
    
    for i in range(0, N+1):
        exactSolution[i] = uExact(k*deltat, i*deltax, ftype)

    approximation = u[k]
    errorNormArr = np.subtract(exactSolution, approximation)
    errorNorm = np.amax(np.abs(errorNormArr))

    return errorNorm

def truncErrorNorm(k, u, T, ftype, met):
    """
    Calculates the norm of the truncation error at the instant t = tk.
    Arguments:
        - k
        - u : 2-dimensional array that stores the temperature at each
          position xi and time tk
        - T : time period
        - ftype : f(x,t) and uExact(t,x) type.
        - met : method used
    """

    M = u.shape[0] - 1
    N = u.shape[1] - 1
    deltax = 1/N
    deltat = T/M

    truncError = np.zeros(N+1)

    # All the calculations start at 1 and finish at N-1. The truncation error cannot be calculated at the boundaries.
    if met == 0 : # Explicit Finite Difference Method
        for i in range(1,N): 
            truncError[i] = (uExact((k+1)*deltat, i*deltax, ftype) - uExact(k*deltat, i*deltax, ftype))/deltat - (uExact(k*deltat, (i-1)*deltax, ftype) - 2*uExact(k*deltat, i*deltax, ftype) + uExact(k*deltat, (i+1)*deltax, ftype))/(deltax**2) - f(k*deltat, i*deltax, ftype)
    elif met == 1 : # Implicit Euler Method
        for i in range(1,N): # Starts at 1 and finishes at N-1. The truncation error cannot be calculated at the boundaries.
            truncError[i] = (uExact((k+1)*deltat, i*deltax, ftype) - uExact(k*deltat, i*deltax, ftype))/deltat - (uExact((k+1)*deltat, (i-1)*deltax, ftype) - 2*uExact((k+1)*deltat, i*deltax, ftype) + uExact((k+1)*deltat, (i+1)*deltax, ftype))/(deltax**2) - f((k+1)*deltat, i*deltax, ftype)
    elif met == 2 : # Crank-Nicolson Method
         for i in range(1,N):
            aux1 = (uExact((k+1)*deltat, i*deltax, ftype) - uExact(k*deltat, i*deltax, ftype))/deltat 
            aux2 = ((uExact((k+1)*deltat, (i-1)*deltax, ftype) - 2*uExact((k+1)*deltat, i*deltax, ftype) + uExact((k+1)*deltat, (i+1)*deltax, ftype)) + (uExact(k*deltat, (i-1)*deltax, ftype) - 2*uExact(k*deltat, i*deltax, ftype) + uExact(k*deltat, (i+1)*deltax, ftype)))/(2*(deltax**2))
            aux3 = (1/2)*(f(k*deltat, i*deltax, ftype) + f((k+1)*deltat, i*deltax, ftype))
            truncError[i] = aux1 - aux2 - aux3

    truncErrorNorm = np.amax(np.abs(truncError))

    return truncErrorNorm    

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

input_list = input("Please input T, N and M respectively, separated by commas: ")
T, N, M = input_list.split(',')

T = int(T)
N = int(N)
M = int (M)

deltax = 1/N
deltat = T/M
lbd = deltat/deltax**2 # Lambda

# Select functions
print()
print("Types :           '0'               |                        '1'                        ")
print("------------------------------------|---------------------------------------------------")
print("f(t,x): 10x^2(x-1) - 60xt + 20t (a1)| 5e^(t - x)*(5t^2*cos(5tx) - sin(5tx)*(x + 2t)) (b)")
print("u0(x) :            0         (a1, c)|                      e^(-x)                    (b)")
print("g1(t) :            0     (a1, a2, c)|                      e^(t)                     (b)")
print("g2(t) :            0     (a1, a2, c)|                      e^(t-1)                   (b)")
print()
print("Types :          '2'          |                          '3'                             ")
print("------------------------------|----------------------------------------------------------")
print("f(t,x): source at p = 0.25 (c)| 10cos(10t)x^2(1-x)^2-(1 + sin(10t))(12x^2 - 12x + 2) (a2)")
print("u0(x) :          N.A.         |                      x^2(1-x)^2                      (a2)")
print("g1(t) :          N.A.         |                         N.A.                             ")
print("g2(t) :          N.A.         |                         N.A.                             ")
print()
input_list = input("Please input f, u0, g1 and g2 types respectively, separated by commas: ")
ftype, u0type, g1type, g2type = input_list.split(',')

ftype = int(ftype)
u0type = int(u0type)
g1type = int(g1type)
g2type = int(g2type)

if (ftype not in [0,1,2,3]) or (u0type not in [0,1,3]) or (g1type not in [0,1]) or (g2type not in [0,1]):
    print()
    print("Invalid input.")

# ---------------
# Defining the grid
# ---------------

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
#tempGraphs(result)

if (ftype != 2):
    # Create learning curve
    errorNorms = np.zeros((M,1))
    truncErrorNorms = np.zeros((M,1))

    bar = Bar("Calculating error norms", max=M)
    for k in range(M):
        errorNorms[k] = errorNorm(k, u, T, ftype)
        truncErrorNorms[k] = truncErrorNorm(k, u, T, ftype, method)
        bar.next()
    bar.finish()

    resErrorNorm = errorNorms[M-1] # Error norm "result". We want the error norm at t = T
    resTruncErrorNorm = np.amax(truncErrorNorms) # Truncation error norm "result". We want the maximum truncation error at all times
    
    print("Absolute error norm at t = T       : ", resErrorNorm)
    print("Truncation error norm at all times : ", resTruncErrorNorm)

    myfile = open("erros.txt", 'a')
    errorString = "Norma do erro absoluto em t = T                 -  N = " + str(N) + " e M = " + str(M) + ": " + str(resErrorNorm) + "\n"
    truncErrorString = "Norma do erro de truncamento em todos os tempos -  N = " + str(N) + " e M = " + str(M) + ": " + str(resTruncErrorNorm) + "\n\n"

    myfile.write(errorString)
    myfile.write(truncErrorString)

    myfile.close()