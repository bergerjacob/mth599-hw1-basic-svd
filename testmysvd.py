import numpy as np
from mysvd import svd
import sys

# Your code only need to work for square matrices to get full credit
n = 10
A = np.random.randn(n,n)

U,S,VT = svd(A)

print()
print("Running Test for 10 x 10 Matrix!")
print()

# Check the shape of U
correct1 = U.shape == (10,10)
print("U.shape correct?",correct1)

# Check the shape of S
correct2 = S.shape == (10,)
print("S.shape correct?",correct2)

# Check the shape of V
correct3 = VT.shape == (10,10)
print("VT.shape correct?",correct3)

if not correct1*correct2*correct3:
    print()
    print("One or more of the shapes of U, S, VT are incorrect!")
    print("Before proceeding fix the shapes of our outputs!")
    print("Grade =  0")
    sys.exit()

# Check U S V = A
err1 = np.linalg.norm(U @ np.diag(S) @ VT - A)/np.linalg.norm(A)
print(err1)

# Check that U is has orthogonal columns
err2 = np.linalg.norm(U.T @ U - np.eye(n))/n
print(err2)

# Check that V has orthogonal columns
err3 = np.linalg.norm(VT @ VT.T - np.eye(n))/n
print(err3)

# Check that S has positive entries
err4 = np.linalg.norm(S - np.abs(S))
print(err4)

# Check that S is sorted in descending order
err5 = np.linalg.norm(S - np.sort(np.abs(S))[::-1])
print(err5)

# Compute Grade
print()
print("Computing Grade")
print()
grade = 0
if err1 < 1e-13:
    grade = grade+40
if err2 < 1e-13:
    grade = grade+20
if err3 < 1e-13:
    grade = grade+20
if err4 < 1e-13:
    grade = grade+10
if err5 < 1e-13:
    grade = grade+10
print("Grade = {}".format(grade))

print()
print("Running Bonus Test")
print()


# For Bonus points!
m,n = 100, 16
A = np.random.randn(m,n)
U1,S1,VT1 = np.linalg.svd(A,full_matrices=False)
S1 = 10**(-np.arange(n,dtype=float))
A = (U1 * S1.reshape(1,-1)) @ VT1

U,S,VT = svd(A)

# Check the shape of U
correct1 = U.shape == (m,n)
print(correct1)

# Check the shape of S
correct2 = S.shape == (n,)
print(correct2)

# Check the shape of V
correct3 = VT.shape == (n,n)
print(correct3)

# Check the dimensions are correct
if not correct1*correct2*correct3:
    print()
    print("One or more of the shapes of U, S, V are incorrect!")
    print("Bonus =  0")
    sys.exit()

# Check U S V = A
err1 = np.linalg.norm((U * S.reshape(1,-1)) @ VT - A)/np.linalg.norm(A)
print(err1)

# Check that U is has orthogonal columns
err2 = np.linalg.norm(U.T @ U - np.eye(n))/n
print(err2)

# Check that V has orthogonal columns
err3 = np.linalg.norm(VT @ VT.T - np.eye(n))/n
print(err3)

# Check that S has positive entries
err4 = np.linalg.norm(S - np.abs(S))
print(err4)

# Check that S is sorted in descending order
err5 = np.linalg.norm(S - np.sort(np.abs(S))[::-1])
print(err5)

# Compute Grade
print()
print("Computing Bonus")
print()

grade = 0
if err1 < 1e-13:
    grade = grade+2
if err2 < 1e-13:
    grade = grade+2
if err3 < 1e-13:
    grade = grade+2
if err4 < 1e-13:
    grade = grade+2
if err5 < 1e-13:
    grade = grade+2
print("Bonus = {}".format(grade))
