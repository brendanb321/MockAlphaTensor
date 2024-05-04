# https://www.nature.com/articles/s41586-022-05172-4

import numpy as np

def main():
    n = 2
    m = 2
    p = 2
    const_T = matrix_multiplication_tensor(n, m, p)
    zeros = np.zeros((n * m, m * p, n * p))
    T = const_T
    Us = [np.array([ 1, 0, 0, 1 ]), 
          np.array([ 0, 0, 1, 1 ]),
          np.array([ 1, 0, 0, 0 ]), 
          np.array([ 0, 0, 0, 1 ]),
          np.array([ 1, 1, 0, 0 ]),
          np.array([-1, 0, 1, 0 ]),
          np.array([ 0, 1, 0,-1 ])]
    Vs = [np.array([ 1, 0, 0, 1 ]),
          np.array([ 1, 0, 0, 0 ]),
          np.array([ 0, 1, 0,-1 ]),
          np.array([-1, 0, 1, 0 ]),
          np.array([ 0, 0, 0, 1 ]),
          np.array([ 1, 1, 0, 0 ]),
          np.array([ 0, 0, 1, 1 ])]
    Ws = [np.array([ 1, 0, 0, 1 ]),
          np.array([ 0, 0, 1,-1 ]),
          np.array([ 0, 1, 0, 1 ]),
          np.array([ 1, 0, 1, 0 ]),
          np.array([-1, 1, 0, 0 ]),
          np.array([ 0, 0, 0, 1 ]),
          np.array([ 1, 0, 0, 0 ])]
    for i in range(len(Us)):
        T = T - np.multiply.outer(np.outer(Us[i], Vs[i]), Ws[i]) # modulo N if desired
    if np.array_equal(T, zeros):
        print(multiplication_formula(Us, Vs, Ws, n, m, p))

def matrix_multiplication_tensor(n, m, p):
    '''
    returns the tensor encoding the matrix multiplication C=AB
    A is nxm, B is mxp, C is nxp
    '''
    tensor = np.zeros((n * m, m * p, n * p))
    for row in range(n):
        for col in range(p):
            for i in range(m):
                tensor[row * m + i][i * p + col][row * p + col] = 1
                # since sum_i A[row][i] * B[i][col] = C[row][col]
    return tensor

def multiplication_formula(Us, Vs, Ws, n, m, p):
    '''
    returns the matrix multiplication formula (as a string) using the vectors Us[i], Vs[i], Ws[i]
    '''
    assert(len(Us) == len(Vs) == len(Ws) > 0)
    assert(len(Us[0]) == n * m and len(Vs[0]) == m * p and len(Ws[0]) == n * p)
    k = len(Us)
    str = f"Multiplication for (n, m, p) = ({n}, {m}, {p}) possible with {k} multiplications\n"
    for i in range(0, k):
        a_formula = extract_formula(Us[i], 'a')
        b_formula = extract_formula(Vs[i], 'b')
        str += f"m_{1+i} = ({a_formula})({b_formula})\n"
    Ws_t = np.array(Ws).T
    for i in range(0, n * p):
        c_formula = extract_formula(Ws_t[i], 'm')
        str += f"c_{1+i} = {c_formula}\n"
    return str

def extract_formula(Xs, char):
    '''
    helper function for print_multiplication_formula
    returns a string representing the expression obtained from the coefficients in Xs
    '''
    formula = ""
    for j in range(0, len(Xs)):
        if Xs[j] == 1:
            formula += f"+{char}_{1+j}"
        elif Xs[j] == -1:
            formula += f"-{char}_{1+j}"
        elif Xs[j] > 0:
            formula += f"+{Xs[j]}{char}_{1+j}"
        elif Xs[j] < 0:
            formula += f"{Xs[j]}{char}_{1+j}"
    if formula[0] == '+':
        return formula[1:]
    return formula

if __name__ == "__main__": 
    main()