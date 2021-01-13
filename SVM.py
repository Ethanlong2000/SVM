import math
from numpy import *


# 选择适当的核函数K(x,z)和一个惩罚系数C>0, 构造约束优化问题
# 用SMO算法求出上式最小时对应的α向量的值α∗向量



class Model:
    def __init__(self,trainSet,list_y):
        self.x=trainSet    #x【i】为list
        self.y=list(list_y)
        self.a = [0] * len(trainSet)   #拉格朗日算子
        self.E = [0] * len(trainSet)  #差值
        self.TrainSetNum=len(trainSet)
        self.C=5      #惩罚系数
        self.b = 0    #阈值
        self.e=0.0001 #KKT条件精度
        self.L=0     #线性约束参数
        self.H=0     #线性约束参数


def K(x,z):  # 高斯核函数（Gaussian Kernel），在SVM中也称为径向基核函数（RBF）
    gama=0.5
    result=[0]*len(x)
    for i in range(len(x)):
        result[i] = (float(x[i]) - float(z[i])) ** 2
    V = math.exp(-gama * (sum(result) ** (1 / 2)))
    return V

def gx(model ,x):   #函数输入为向量x,输出为值
    value=0

    for n in range(model.TrainSetNum):

        value=value+(model.a[n])*(model.y[n])*K(model.x[n],x)
    finalvalue=value+model.b
    return finalvalue
# 求第一个变量  外层循环

def get_alpha1(model): #计算第一个拉格朗日参数alpha1(此处为了方便，将其值和序号放一起，作为list输出)
    alpha1=[0,0,0]     # new值，index，old值
    for i in range(model.TrainSetNum):
        if model.a[i]==0 and (model.y[i])*gx(model,model.x[i])<1:
            alpha1[2]=alpha1[0]
            alpha1[0]=model.a[i]
            alpha1[1]=i
            break
        elif model.a[i]>0 and model.a[i]<model.C and abs((model.y[i])*gx(model,model.x[i])-1)>model.e:

            alpha1[2] = alpha1[0]
            alpha1[0] = model.a[i]
            alpha1[1] = i
            break
        elif model.a[i]==model.C and (model.y[i])*gx(model,model.x[i])>1:
            alpha1[2] = alpha1[0]
            alpha1[0] = model.a[i]
            alpha1[1] = i
            break
    return alpha1

def Get_E(model,index):
    E= gx(model, model.x[index]) - (model.y[index])
    return E

def set_E1(model,alpha1):
    index=alpha1[1]
    model.E[index]=Get_E(model,index)

# 求第二个变量 内层循环

def get_alpha2(model,alpha1):  #计算alpha2
    alist=[0]*len(model.a)
    alist2 = [0] * len(model.a)
    ylist=[0]*len(model.a)
    xlist=[0]*len(model.a)
    for i in range(len(model.a)):
        alist[i] = model.a[i]
        alist2[i] = model.a[i]
        ylist[i]=model.y[i]
        xlist[i]=model.x[i]
    alist.pop(alpha1[1])  #将alpha1排除
    ylist.pop(alpha1[1])
    xlist.pop(alpha1[1])
    alist2[alpha1[1]]=666#避免两个alpha取到相同index
    Elist=[0]*len(alist)
    alpha2=[0,0,0]    # new值，index，old值
    E1=Get_E(model,alpha1[1])
    for j in range(len(alist)):
        value=0
        for n in range(len( alist)):
            value = value + (alist[n]) * (ylist[n]) * K(xlist[n], xlist[j])
        gx= value + model.b
        Elist[j]=gx-ylist[j]

    if E1>0:
          alpha2[2]=alpha2[0]
          alpha2[0]=alist[Elist.index(min(Elist))]
          alpha2[1]=alist2.index(alpha2[0])
          return alpha2
    else:
          alpha2[2] = alpha2[0]
          alpha2[0] = alist[Elist.index(max(Elist))]
          alpha2[1] = alist2.index(alpha2[0])
          return alpha2

#更新边界
def  update_L_H(model,alpha1,alpha2):
    index1 = alpha1[1]
    index2 = alpha2[1]
    y1=model.y[index1]
    y2=model.y[index2]
    if y1!=y2:
        model.L=max(0, alpha2[2]-alpha1[2])
        model.H=min(model.C, model.C+alpha2[2]-alpha1[2])
    else:
        model.L = max(0, alpha2[2] + alpha1[2]-model.C)
        model.H = min(model.C,  alpha2[2] + alpha1[2])

def Get_a_newunc(model,alpha2,alpha1): #求更新alpha2的一个中间参数，即未限制的alpha2new
    # a=alpha1[1]
    # b=alpha2[1]
    # c=a+b
    a_newunc=alpha2[2]+alpha2[1]*(Get_E(model,alpha1[1])-Get_E(model,alpha2[1]))/(1+1-2*K(model.x[alpha1[1]],model.x[alpha2[1]])) #k()里相同，exp（0）=1
    return a_newunc

def Getalpha2_new(model,a_newunc):
    alpha2_new=0
    if a_newunc>model.H:
        alpha2_new=model.H
    elif a_newunc<model.L:
        alpha2_new=model.L
    elif a_newunc>=model.L and a_newunc<=model.H:
        alpha2_new=a_newunc
        
    return alpha2_new

def Getalpha1_new(model,alpha1,alpha2): #先更新alpha2，再更新alpha1
    alpha1_new=alpha1[0]+(model.y[alpha1[1]])*(model.y[alpha2[1]])*(alpha2[2]-alpha2[0])
    return alpha1_new
#先更新两个alpha，再跟新list

def Update_alphalist(model,alpha1,alpha2):
    model.a[alpha1[1]] = alpha1[0]
    model.a[alpha2[1]] = alpha2[0]

# 计算阈值b和差值Ei
def Update_b(model,alpha1,alpha2):
    index1=alpha1[1]
    index2=alpha2[1]
    b1_new=-Get_E(model,index1)+(alpha1[2])*(model.y[index1])*1+alpha2[2]*(model.y[index2])*K(model.x[index2],model.x[index1])+model.b-alpha1[0]*(model.y[index1])*1-alpha2[0]*(model.y[index2])*K(model.x[index2],model.x[index1])
    b2_new=-Get_E(model,index2)-(model.y[index1])*K(model.x[index1],model.x[index2])*(alpha1[0]-alpha1[2])-(model.y[index2])*1*(alpha2[0]-alpha2[2])+model.b
    model.b=(b1_new+b2_new)/2


def Update_Ei(model):
    index=[]
    for i in range(model.TrainSetNum):
        if model.a[i]>0:
            index.append(i)
    for i in range(model.TrainSetNum):
        E=0
        for j in range(len(index)):
            E+=model.y[j]*model.a[j]*K(model.x[i],model.x[j])
        E+=model.b-model.y[i]
        model.E[i]=E

def Check(model):
    END=True

    for index in range(model.TrainSetNum):
        if model.a[index]<0 or model.a[index]>model.C:
            END=False
            break
        elif model.a[index] ==0 and  (model.y[index])*gx(model,model.x[index])<1:
            END=False
            break
        elif model.a[index] >0and model.a[index]<model.C and abs(model.y[index]*gx(model,model.x[index])-1)>model.e:
            END=False
            break
        elif model.a[index] ==model.C and model.y[index] * gx(model, model.x[index]) > 1:
            END = False
            break
    return END

def check_alpha(alpha1,alpha2):#如果alpha的值几乎不变，跳出循环
    precision=0.00000001
    if abs(alpha1[0]-alpha1[2])<precision and abs(alpha2[0]-alpha2[2])<precision:
        return True
    else:
        return False

def SMO(model):
    END=False
    iteratnum = 0
    MAX_iteration=100
    while((~END) and (iteratnum<MAX_iteration)):

        alpha1=get_alpha1(model)
        set_E1(model, alpha1)
        alpha2=get_alpha2(model,alpha1)
        update_L_H(model, alpha1, alpha2)
        a_newunc=Get_a_newunc(model, alpha2, alpha1)
        alpha2_new=Getalpha2_new(model, a_newunc)
        alpha2[2]=alpha2[0]
        alpha2[0]=alpha2_new  #更新alpha2
        alpha1_new=Getalpha1_new(model, alpha1, alpha2)
        alpha1[2] = alpha1[0]
        alpha1[0] = alpha1_new  # 更新alpha1
        Update_alphalist(model, alpha1, alpha2)
        Update_b(model, alpha1, alpha2)
        Update_Ei(model)
        END1=Check(model)
        END2=check_alpha(alpha1,alpha2)
        # END=END1 or END2
        END=END2
        iteratnum += 1
    return model

def Get_fx(model):
    sv = []    #支持向量的集合
    for i in range(model.TrainSetNum):
        if model.a[i] > 0:
            sv.append(i)
    bs=[]
    for j in range(len(sv)):
        v = 0
        for i in range(model.TrainSetNum):
            v +=model.y[j]- model.y[i] * model.a[i] * K(model.x[i], model.x[j])
        bs.append(v)
    model.b=mean(bs) #用平均值做最终的b

def fx(model,testset):
    result=[]
    # values=[]
    for i in range(len(testset)):
        sum=0
        for j in range(model.TrainSetNum):
            sum+=model.y[j] * model.a[j] * K(model.x[i], model.x[j])
        # values.append(sum+model.b)
        if (sum+model.b)>0:
            result.append(1)
        else:
            result.append(-1)
    return result
    # return values