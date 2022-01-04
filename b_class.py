import numpy as np
from cvxopt import matrix, solvers
import sys
import math

def readData(fname):
    a = np.genfromtxt(fname, delimiter=',')
    m,n = np.shape(a)
    # print("Data shape is:",m,n)
    
    d1 = a[a[:,n-1]==8.0]
    # print(np.shape(d1))
    d2 = a[a[:,n-1]==9.0]
    # print(np.shape(d2))
    data = np.vstack((d1,d2))
    m,n = np.shape(data)
    # print(np.shape(data))
    x = data[0:m,0:n-1]
    x = x/255
    y = data[0:m,n-1:n]
    m1,n1 = np.shape(x)
    for i in range(m1):
        if y[i][0] == 8.0:
            y[i][0]=-1.0
        else:
            y[i][0]=1.0


    # print("x shape:",np.shape(x))
    # print("y shape:",np.shape(y))
    return x,y

def gaussianKernal(x1,x2):
    # print(np.shape(x1),np.shape(x2))
    a = x1 - x2
    # print(a)
    ans = np.dot(a,a)
    # print(ans)
    # ans = ans[0][0]
    ans = ans*-0.05
    ans = math.exp(ans)
    # print(ans)
    return ans

def linearKernal(x1,x2):
    ans = np.dot(x1,x2)
    return ans

def main(mod):
    kernal = None
    if mod == 0:
        kernal = linearKernal
    if mod == 1:
        kernal = gaussianKernal
    # read data, last digit of entry no is 8
    x,y = readData(sys.argv[1])
    
    m,n = np.shape(x)
    c = 1.0         #hyper parameter
    q = -1*np.ones(m)
    G = np.vstack((-1*np.identity(m), np.identity(m)))
    # print("G shape is:",np.shape(G))
    h = np.append(np.zeros(m),c*np.ones(m))
    # print("h shape is:",np.shape(h))
    A = np.transpose(y)
    b = np.zeros(1)
    # x=np.transpose(x)
    # p = (x.T @ x)*(y.T @ y)
    # print(np.shape(p))
    # x=np.transpose(x)

    k = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            k[i][j] = kernal(x[i],x[j])
    
    p = np.outer(y,y)*k
                

    # print("coeffs found")
    # print(p)
    # p = np.matmul(p,np.transpose(p))
    
    # print("p shape is:",np.shape(p))        
    
    # forming arguments
    p = matrix(p)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    solvers.options['show_progress'] = False
    sol = solvers.qp(p,q,G,h,A,b)
    alpha = sol['x']
    # print("objective",sol['dual objective'])
    # finding w
    sv = []   # set of support vectors
    
    for i in range(m):
        if alpha[i]<c and alpha[i]>0:
            sv.append(i)

    # finding b = -min_{i,y[i]=1}(wTx[i])-max_{i,y[i]=-1}(wTx[i])  / 2 x[i] are sv
    mi = float("inf")
    mx = float("-inf")
    # print(np.shape(x))
    # temp  = np.matmul(w,np.transpose(x))

    # print(np.shape(temp))
    
    for i in range(len(sv)):
        ans =0
        for j in range(m):
            ans += alpha[j]*y[j][0]*k[sv[i]][j]

        if y[sv[i]]>0:
            mi = min(mi,ans)
        else:
            mx = max(mx,ans)
        
    # print(mi)
    # print(mx)
    b = (mi+mx)/2
    b=b*-1
    # print(b)



    # predict on val data
    # x1,y1 = readData('val.csv')
    # # print(x1,y1)
    # m1,n1 = np.shape(x1)
    # val_score = 0
    # for i in range(m1):
    #     ans =0
    #     for j in range(m):
    #         ans += alpha[j]*y[j][0]*kernal(x[j],x1[i])
    #     ans+=b
    #     if ans*y1[i][0] >0 :
    #         val_score+=1


    # print("val acuracy is:", val_score/m1)
    output = open(sys.argv[3], 'w')
    # x2,y2 = readData(sys.argv[2])
    a = np.genfromtxt(sys.argv[2], delimiter=',')
    m2,n2 = np.shape(a)
    x2,y2 = a[0:m2,0:n2-1],a[0:m2,n2-1:n2]
    x2 = x2/255
    m2,n2 = np.shape(x2)
    test_score = 0
    for i in range(m2):
        ans =0
        for j in range(m):
            ans += alpha[j]*y[j][0]*kernal(x[j],x2[i])
        ans+=b
        if ans*y2[i][0] >0 :
            test_score +=1
        if ans > 0:
            print("9",file=output)
        else:
            print("8",file=output)
    
    # print("test acuracy is:", test_score/m2)
        

# Results linear kernel
# x shape: (4500, 784)
# y shape: (4500, 1)
# train acuracy is: 1.0
# x shape: (500, 784)
# y shape: (500, 1)
# val acuracy is: 0.996
# x shape: (1000, 784)
# y shape: (1000, 1)
# test acuracy is: 0.999

# Results gaussian kernel
# (4500, 785)
# x shape: (4500, 784)
# y shape: (4500, 1)
# coeffs found
# -0.09955544992405674
# train acuracy is: 1.0
# (500, 785)
# x shape: (500, 784)
# y shape: (500, 1)
# val acuracy is: 0.996
# (1000, 785)
# x shape: (1000, 784)
# y shape: (1000, 1)
# test acuracy is: 0.999

# results for d=3 gaussian
# (4500, 785)
# x shape: (4500, 784)
# y shape: (4500, 1)
# coeffs found
# -1.0458069273152888
# train acuracy is: 0.9862222222222222
# (500, 785)
# x shape: (500, 784)
# y shape: (500, 1)
# val acuracy is: 0.936
# (1000, 785)
# x shape: (1000, 784)
# y shape: (1000, 1)
# test acuracy is: 0.945

# print(sys.argv[1])
main(1)
# gaussianKernal(np.array([1,2,3]),np.array([1,2,3]))
