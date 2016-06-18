
#coding=utf-8
__author__ = 'shuai'
import sys
import numpy as np
import random
import re
import matplotlib.pyplot as plt

reload(sys)
sys.setdefaultencoding('utf-8')

def logisitic(x):
    return 1/(1+np.exp(-x))
def activeFunction(x):
    if logisitic(x)>=0.5:
        return 1
    return 0
#logistic regression也可以称为感知器网络，为了处理线性可分的情况
#f(x)=wxT+b,w为权重，b为偏向
#logistic 公式为：1/(1+exp(-wTx+b))
class LogisticRegression:
    def __init__(self,esp=0.0001,alf=0.005,steps=500):
        self.esp=esp
        self.alf=alf
        self.steps=steps
        self.X=[]
        self.Y=[]
        self.weights=[]
        self.bayes=0

    #训练数据，数据格式为(X,Y)包含labels
    def train(self,data):
        self.X=np.mat(data).T[:-1].T# 去除labels
        self.Y=np.mat(data).T[-1].T# 获得labels
        self.weights=np.zeros([len(data[0])-1,1])
        #训练数据
        #wk+1=wk+alf*error*xT
        for num in range(self.steps):
            #算出计算值，即wx+b
            target=self.X*self.weights+np.ones([len(self.X),1])*self.bayes
            #根据计算值算出分类值根据激活函数logistic
            output=map(activeFunction,target)
            output=np.mat(output).T
            #计算出误差
            error=self.Y-output
            #更新权重
            print 'self.Y',self.Y
            print 'error',error
            self.weights=self.weights+self.X.T*self.alf*error
            self.bayes=self.bayes+self.Y.T*self.alf*error
            #if(sum(error<self.esp)):
                #break

    def predict(self,x):
        logisiticVaue=logisitic(np.mat(x)*self.weights+self.bayes)
        print logisiticVaue,logisiticVaue
        return activeFunction(logisiticVaue)

    def calPrecisionRate(self,data):
        precisionNum=0
        for index in range(len(data)):
            predictValue=self.predict(data[index][:-1])
            #预测值和真实值差绝对值如果为0，正确，如果为1，不正确
            precisionNum+=abs(predictValue-data[index][-1])
        return predictValue/len(data)

if __name__=='__main__':
    file=open(r'C:\Users\shuai\Desktop\logistic.txt','rb')
    data=[]
    x=[]
    y=[]
    for line in file.readlines():
        item=re.split('\s+',line.strip())
        item=map(lambda x:float(x),item)
        x.append(item[:-1])
        y.append(item[-1])
        data.append(item)
    print data
    print len(x)
    print len(y)
    plt.figure(1)
    plt.subplot(111)
    for index in range(len(y)):
        if y[index]==1:
            plt.scatter(x[index][0],x[index][1],c='red',marker='o')
        else:
            plt.scatter(x[index][0],x[index][1],c='blue',marker='s')
    logisticRegression=LogisticRegression()
    logisticRegression.train(data)
    print logisticRegression.weights,logisticRegression.bayes
    #y=ax+b
    b=-logisticRegression.bayes/logisticRegression.weights[1]
    a=-logisticRegression.weights[0]/logisticRegression.weights[1]
    xa=np.linspace(-5,5,100)
    ya=-(float(logisticRegression.bayes)+xa*(float(logisticRegression.weights[0])))/float(logisticRegression.weights[1])
    print xa,ya

    #使用sklearn




    plt.plot(xa,ya)
    plt.show()
