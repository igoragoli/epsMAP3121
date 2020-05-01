# EP1 - MAP3121

## To-Do

### Functions and Methods

1. [x] LU Decomposition function
2. [x]  LU Solver function
3. [x] Permutation Matrix function
4. [x] Check for singularity in LUDecomposition()
5. [x] Implement Implicit Euler Method
6. [x] Implement Crank-Nicholson Method
7. [x] LDL-1 Decomposition function
8. [x] Change solveLinearSystem() to use LDL-1 decomp. instead of LU decomp
9. [x] Deduce f, u0, g1, g2 for itens (a), (b) and (c) and complete their functions
10. [x] Read all the fucking program and correct x <--> t for wrong functions

### Simulations

1. Primeira Tarefa
    Deve-se simular com $N = 10, 20, 40, 80, 160, 320, 640$, e $T = 1$.
    a1) Para funções do item a1:
        - [] Simular para $\lambda = 0.25$, $\lambda = 0.5$ e $\lambda = 0.51$. O que muda entre os casos? 
        - [] Calcular o erro para cada $\lambda$. Qual é o comportamento do erro?
        - [] Calcular o erro em T = 1 (deve ser nulo). 
        - [] Verificar que a solução exata é $10tx^2(x-1)$
        - [] Qual é o fator de redução esperado a cada refinamento de malha?
        - [] Qual o número de passos necessários ao se usar $N = 640$? E se dobrarmos N?
    a2) Para funções do item a2, repetir o item a1):
        - [] Repetir os experimentos de a1[]
    b) Para funções do item b:
        - [x] Determinar as funções das condições de contorno.
        - [] Repetir os experimentos de a1)
    c) Para funções do item c:
        - [x] Determinar f.
        - [] Repetir os experimentos de a1)
     
2. Segunda Tarefa
    a) Escreva um procedimento que efetue a decomposicao $LDL^t$
        - [x] Escrever LDLtDecomposition(). 
    Preparação para próximos itens: 
        - [x] Escrever eulerImplicitMethod().
        - [x] Determinar formula matricial de Crank Nicholson e escrever crankNicholson().
3. Test crankNicholson() on "Um metodo implicito"

### Questions

1. Should we substitute b = np.zeros((N-1,1)) in implicitEuler() and in 
crankNicholson() for b = np.zeros(N-1)? 
