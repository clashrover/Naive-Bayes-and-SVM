import numpy as np
from cvxopt import matrix, solvers
import sys
import math

def splitData(a):
    l=[]
    for i in range(10):
        m,n = np.shape(a)
        d = a[a[:,n-1]==float(i)]
        m,n = np.shape(d)
        x = d[0:m,0:n-1]
        x = x/255
        y = d[0:m,n-1:n]
        l.append((x,y))
    return l

def gaussianKernal(x1,x2):
    a = x1 - x2
    ans = np.dot(a,a)
    ans = ans*-0.05
    ans = math.exp(ans)
    return ans

def linearKernal(x1,x2):
    ans = np.dot(x1,x2)
    return ans



def train_classifier(ic,jc,data1,data2,kernal = gaussianKernal):
    
    # print(data1)
    # print(data2)
    x1,y1 = data1
    m1,n1 = np.shape(x1)
    x2,y2 = data2
    m2,n2 = np.shape(x2)
    x = np.vstack((x1,x2))
    y = np.vstack((y1,y2))
    m,n = np.shape(x)
    for i1 in range(m):
        if y[i1][0]==ic:
            y[i1][0]=-1
        else:
            y[i1][0]=1
    
    c = 1.0         #hyper parameter
    q = -1*np.ones(m)
    G = np.vstack((-1*np.identity(m), np.identity(m)))
    h = np.append(np.zeros(m),c*np.ones(m))
    A = np.transpose(y)
    b = np.zeros(1)

    xxt = np.matmul(x,np.transpose(x))
    xd = np.diag(xxt)
    
    xr = np.tile(xd,(m,1))
    xc = np.transpose(xr)

    k = xc-2*xxt + xr
    k = np.exp(-0.05*k)
    # k = (np.diag(x@x.T) colum broadcasting) - (2*x@x.T) + (np.diag(x@x.T) row broadcasting)
    # k = np.exp(-0.05*k)

    p = np.outer(y,y)*k  
    
    # forming arguments
    p = matrix(p)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    solvers.options['show_progress'] = False
    print("arguments formed")
    sol = solvers.qp(p,q,G,h,A,b)
    print("cvxopt solved")
    alpha = sol['x']

    sv = []   # set of support vectors
    
    for i in range(m):
        if alpha[i]<c and alpha[i]>0:
            sv.append(i)

    # finding b = -min_{i,y[i]=1}(wTx[i])-max_{i,y[i]=-1}(wTx[i])  / 2          -> where x[i] are sv
    mi = float("inf")
    mx = float("-inf")
    
    # k = np.transpose(k)
    for i in range(len(sv)):
        ans=0
        for j in range(m):
            ans += alpha[j]*y[j][0]*k[sv[i]][j]
        if y[sv[i]][0]>0:
            mi = min(mi,ans)
        else:
            mx = max(mx,ans)
        
    b = (mi+mx)/2
    b=b*-1
    print("classifier calculated:",ic,jc)
    return alpha, b, y, x


def predict(e, classifier, kernal = gaussianKernal):
    alpha,b,y,x = classifier
    m,n = np.shape(x)
    xet = np.matmul(x,np.transpose(e))
    xxt = np.matmul(x,np.transpose(x))
    eet = np.matmul(e,np.transpose(e))
    m1,n1 = np.shape(e)
    ans=[0]*m1
    for i in range(m1):
        for j in range(m):
            ans += alpha[j]*y[j][0]*math.exp(-0.05*(xxt[j][j]+eet[i][i]-(2*xet[j][i])))
        ans[i]+=b

    return ans




def main():
    a = np.genfromtxt('train.csv', delimiter=',')
    m,n = np.shape(a)
    
    l = splitData(a)
    
    classifiers = []
    for i in range(9):
        for j in range(i+1,10):
            classifiers.append(train_classifier(float(i),float(j),l[i],l[j]))
            # return
    print("classifiers obtained")
    return
    # # predict on train set
    # train_score=0
    # for k in range(10):
    #     x = l[k][0]     # all examples of class k
    #     m,n = np.shape(x)
    #     for i in range(m):
    #         eg = x[i]
    #         score = [0]*10  # to track which class it matches maximum times
    #         distance = [0]*10
    #         k1=0
    #         for i1 in range(9):
    #             for j1 in range(i+1,10):
    #                 label, dis = predict(eg,classifiers[k1])
    #                 if label<0:
    #                     score[i1]+=1
    #                     distance[i1] = max(distance[i1],abs(dis))
    #                 else:
    #                     score[j1]+=1
    #                     distance[j1] = max(distance[j1],abs(dis))
    #                 k1+=1
    #         mx1=-1
    #         disx=0
    #         class_label=-1
    #         for i1 in range(10):
    #             if score[i1]>mx1:
    #                 mx1 = score[i1]
    #                 disx = distance[i1]
    #                 class_label=i1
    #             elif score[i1]==mx1:
    #                 if disx<distance[i1]:
    #                     disx = distance[i1]
    #                     class_label=i1
    #         if class_label == k:
    #             train_score+=1

    print("predict on val")
    a1 = np.genfromtxt('val.csv', delimiter=',')
    m1,n1 = np.shape(a1)
    
    l1 = splitData(a1)
    
    val_score=0
    for k in range(10):
        x1 = l1[k][0]     # all examples of class k
        y1 = l1[k][1]
        m1,n1 = np.shape(x1)
        k1=0
        scoreList = np.zeros((m1,10))
        distanceList = np.zeros((m1,10))
        for i in range(9):
            for j in range(i+1,10):
                labels = predict(x1,classifiers[k1])
                for i1 in range(m1):
                    if labels[i1]<0:
                        scoreList[i1][i]+=1
                        distanceList[i1][i] = max(distanceList[i1][i], abs(labels[i1]))
                    else:
                        scoreList[i1][j]+=1
                        distanceList[i1][j] = max(distanceList[i1][j], abs(labels[i1]))

    
        for i in range(m1):
            mi=-1
            mx=0
            md=0
            for j in range(10):
                if mx<scoreList[i][j]:
                    mx=scoreList[i][j]
                    mi=j
                    md = distanceList[i][j]
                elif mx==scoreList[i][j]:
                    if md<distanceList[i][j]:
                        mi=j
                        md = distanceList[i][j]
            if mi == y[i][0]:
                val_score+=1
    
    # print(val_score/m1)


            
        # for i in range(m1):
        #     eg = x1[i]
        #     score = [0]*10  # to track which class it matches maximum times
        #     distance = [0.0]*10
        #     k1=0
        #     for i1 in range(9):
        #         for j1 in range(i+1,10):
        #             dis = predict(eg,classifiers[k1])
        #             if dis<0:
        #                 score[i1]+=1
        #                 distance[i1] = max(distance[i1],abs(dis))
        #             else:
        #                 score[j1]+=1
        #                 distance[j1] = max(distance[j1],abs(dis))
        #             k1+=1
        #     mx1=-1
        #     disx=0.0
        #     class_label=-1
        #     for i1 in range(10):
        #         if score[i1]>mx1:
        #             mx1 = score[i1]
        #             disx = distance[i1]
        #             class_label=i1
        #         elif score[i1]==mx1:
        #             if disx<distance[i1]:
        #                 disx = distance[i1]
        #                 class_label=i1
        #     if class_label == k:
        #         val_score+=1
    
    m1,n1 = np.shape(a1)
    print("val acuracy:", val_score/m1)

    # print("predict on test")
    # a2 = np.genfromtxt('test.csv', delimiter=',')
    # m2,n2 = np.shape(a2)
    
    # l2 = splitData(a2)
    
    # test_score=0
    # for k in range(10):
    #     x2 = l2[k][0]     # all examples of class k
    #     m2,n2 = np.shape(x1)
    #     for i in range(m2):
    #         eg = x2[i]
    #         score = [0]*10  # to track which class it matches maximum times
    #         distance = [0]*10
    #         k1=0
    #         for i1 in range(9):
    #             for j1 in range(i+1,10):
    #                 dis = predict(eg,classifiers[k1])
    #                 if dis<0:
    #                     score[i1]+=1
    #                     distance[i1] = max(distance[i1],abs(dis))
    #                 else:
    #                     score[j1]+=1
    #                     distance[j1] = max(distance[j1],abs(dis))
    #                 k1+=1
    #         mx1=-1
    #         disx=0
    #         class_label=-1
    #         for i1 in range(10):
    #             if score[i1]>mx1:
    #                 mx1 = score[i1]
    #                 disx = distance[i1]
    #                 class_label=i1
    #             elif score[i1]==mx1:
    #                 if disx<distance[i1]:
    #                     disx = distance[i1]
    #                     class_label=i1
    #         if class_label == k:
    #             test_score+=1
    
    # m2,n2 = np.shape(a2)
    # print("test acuracy:", test_score/m1)

    
    


main()