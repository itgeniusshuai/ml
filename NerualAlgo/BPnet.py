#coding=utf-8
__author__ = 'shuai'
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


class BPnet:
    def __init__(self):
        self.eb=0.01                #误差容限，当误差小于他时算法收敛
        self.iterator=0             #算法收敛式的迭代次数
        self.eta=0.1                #学习率
        self.mc=0.3                 #动量因子，调优参数
        self.maxiter=2000           #最大迭代次数
        self.nHidden=4              #隐藏层个数
        self.nOut=1                 #输出层个数
        self.errlist=[]             #误差列表，保存了误差变换，用于评定收敛
        self.dataMat=0              #训练集
        self.classLabels=0          #分类标签集
        self.nSameNum=0             #样本集行数
        self.nSampDim=0             #样本列数