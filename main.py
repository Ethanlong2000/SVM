import csv
import  SVM
import numpy as np
import random
import math
from collections import Counter

#读文件，生成训练集、测试集
def listequal(list):
    copylist=[]
    for element in list:
        copylist.append(element)
    return copylist
def deal_traindata(trainset,list_y,list_x,STARTNUM,ROW1,ROW2,STOPNUM):
    for i in range(STARTNUM,ROW1):
        list_y[i-STARTNUM]=trainset[i][-1]
        x=[]
        for element in trainset[i]:
            x.append(element)
        del x[-1]
        list_x[i-STARTNUM]=listequal(x)
    for j in range(ROW2, STOPNUM):
        list_y[j-ROW2+ROW1-STARTNUM]=trainset[j][-1]
        y = []
        for element in trainset[j]:
            y.append(element)
        del y[-1]
        list_x[j-ROW2+ROW1-STARTNUM]=listequal(y)


def deal_testdata(trainSet,list_y,list_x,ROW1,ROW2):
    for i in range(ROW1,ROW2):
        list_y[i-ROW1]=trainSet[i][-1]
        x=[]
        for element in trainSet[i]:
            x.append(element)
        del x[-1]
        list_x[i-ROW1]=x

def DisruptOrder(lst):
            l = len(lst)
            if l <= 1:
                return lst
            i = 0
            while l > 1:
                p = int(random.random() * l)
                lst[i], lst[i + p] = lst[i + p], lst[i]
                i += 1
                l -= 1
            return lst

STARTNUM=0
STOPNUM=102
SPAN=int((STOPNUM-STARTNUM)/10)
FOLD_NUM=0
with open('dataset.csv', 'r') as f:
    reader = csv.reader(f)
    trainSet = list(reader)
DisruptOrder(trainSet)

#训练阶段
while FOLD_NUM<10:
    ROW1=1+FOLD_NUM*SPAN
    ROW2=ROW1+SPAN

    trainlist_x={}
    trainlist_y={}
    print (STARTNUM,ROW1,ROW2,STOPNUM)
    deal_traindata(trainSet,trainlist_y,trainlist_x,STARTNUM,ROW1,ROW2,STOPNUM)
    model=SVM.Model(trainlist_x,trainlist_y)
    model=SVM.SMO(model)
    SVM.Get_fx(model)

    #测试阶段
    testlist_x= {}
    testlist_y={}
    deal_testdata(trainSet,testlist_y,testlist_x,ROW1,ROW2)
    testresult=listequal(SVM.fx(model,testlist_x))
    # print(result)
    P_result=[]
    N_result=[]
    for i in range(len(testlist_y)):
        if float(testlist_y[i])>0:
            if testresult[i]==float(testlist_y[i]):
                P_result.append(1)    #样本为正，结果为正
            else:
                P_result.append(0)
        else:
            if testresult[i]==float(testlist_y[i]):
                N_result.append(1)   #样本为负，结果为负
            else:
                N_result.append(0)
    FP = sum(P_result)
    TP = len(N_result)-sum(N_result)
    FN =sum(N_result)
    TN =len(P_result)-sum(P_result)
    accuracy=(TP+TN)/(TP+FP+TN+FN)
    if (TP+FP)>0:   #考虑某次样本全为负
     precision=TP/(TP+FP)
    else:
        precision =0
    if (TP+FN)>0:
        recall=TP/(TP+FN)#考虑某次全为正
    else:
        recall=0
    F_measure=(2*accuracy*precision)/(precision+accuracy)
    print("第", (FOLD_NUM + 1), "折验证的结果为:", '\n'
            "准确率为：", accuracy, '\n'
            "精准率为：", precision, '\n'
            "召回率为：", recall, '\n'
            "F1值为：", F_measure
          )
    FOLD_NUM+=1