import numpy as np
from scipy.sparse import coo_matrix

def jacobirot(A,i,j):

    assert A.shape[0]==A.shape[1]
    n = A.shape[0]
    
    a = A[i,i]
    b = A[i,j]
    c = A[j,i]
    d = A[j,j]
    
    theta, phi = compang(a,b,c,d)
    
    Jtheta = sparserot(n,i,j,theta)
    Jphi= sparserot(n,i,j,phi)

    return Jtheta, Jphi



def sparserot(n,i,j,theta):
    rows = np.arange(n+2)
    cols = np.arange(n+2)
    values = np.ones(n+2)
    
    values[i] = np.cos(theta)
    values[j] = np.cos(theta)
    values[n] = -np.sin(theta)
    values[n+1] = np.sin(theta)

    rows[n] = i
    cols[n] = j
    
    rows[n+1] = j
    cols[n+1] = i

    return coo_matrix((values, (rows, cols)), shape=(n, n))



def compang(a,b,c,d):
    x = np.atan2(b-c,a+d)
    y = np.atan2(b+c,a-d)
    return (x-y)/2, (x+y)/2