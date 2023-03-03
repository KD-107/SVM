import numpy as np
import pandas as pd
from icecream import ic
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import make_blobs

def entropy(data_df,columns):
    pe_value_array = data_df[columns].unique()
    ent = 0.0
    for x_value in pe_value_array:
        p = float(data_df[data_df[columns] == x_value].shape[0]) / data_df.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent

def draw_label(x):
    label = np.array(x)
    label_list = [sum(label)[0], len(label) - sum(label)[0]]
    plt.pie(label_list,autopct='%.2f%%',labels=['REPEAT = 1','REPEAT = 0'])
    plt.savefig("./picture/draw_label.png")

def draw_feature(x,cloumns):
    saveplace = './picture/features/'+str(cloumns)+'.png'
    ic(saveplace)
    sort_list = x[cloumns].sort_values(ascending=False)
    # ic(sort_list)
    plt.title(str(cloumns))
    plt.bar(x=range(len(x[cloumns])),height=sort_list)
    plt.savefig(saveplace)

def select_feature(x,y):
    x_new = SelectKBest(k = 40)
    x_new_feature = x_new.fit_transform(x, y)
    score = x_new.scores_
    indices = np.argsort(score)[::-1]
    k_best_list = []

    for i in range(40):
        k_best_feature = x.columns[indices[i]]
        k_best_list.append(k_best_feature)

    return x_new_feature,k_best_list

def corr(x,y):
    matrix = pd.concat([x,y],axis=1)
    cor = matrix.corr(method='pearson')
    return cor

def ske_kue(x,cloumns):
    s = pd.Series(x[cloumns])  # 将列表x转换为pandas中的Series，其实就相当于一维的矩阵
    print('偏度  = ', s.skew(), '峰度:', s.kurt())  # 计算偏度和峰度
    return s.skew(),s.kurt()

def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return x

def SVM(x,cloumns,y):
    modelSVM = SVC(kernel='linear', C=100)
    print("training")
    modelSVM.fit(x[cloumns],y)
    ic(modelSVM.intercept_,modelSVM.coef_)
    svm_f_1 = open('./data/svm_intercept.txt','w')
    svm_f_2 = open('./data/svm_coef.txt','w')
    for value_1 in modelSVM.intercept_:
        svm_f_1.write(float(value_1))
    for value_2 in modelSVM.coef_:
        svm_f_2.write(float(value_2))
    svm_f_1.close()
    svm_f_2.close()


def main():
    data = pd.read_csv("E:\OnlineJudge\DataAnsys\Lab3\data\processed_data.csv")

    label = pd.read_csv("E:\OnlineJudge\DataAnsys\Lab3\data\label.csv")
    pica = pd.read_csv("E:\OnlineJudge\DataAnsys\Lab3\data\pica2015.csv",low_memory=False)
    new_key = pica.columns[245:len(pica.columns)-1]
    new_data = data[new_key]

    new_pica,best40_feature_list = select_feature(new_data,label)
    ic(new_pica)
    ic(best40_feature_list)

    # 分类
    SVM(pica,best40_feature_list,label['REPEAT'])
    # 下面的数据(coef,intercept)是由于SVM一次训练时间过长,此处拿其中一次训练结果为例进行预测
    coef = 15529.59226939
    intercept = np.array([-5.68105005e+00, -6.37708844e+00, -6.04235713e+00,-5.50950198e+00, -5.21255654e+00, -2.17019701e+00,-4.22719605e+00, -4.95479307e+00, -2.48543769e+00,-2.33734947e+00, -1.35756263e+00, -2.66418745e+00,-1.00168975e+00, -3.44009636e+00,  8.67477581e-01,-2.22037058e+00, -3.88820463e+00, -1.24632849e+00,9.39162970e-02, -9.57975730e-01, -3.32886854e+00,1.19442402e+00,  1.66210979e-01,  2.83926342e+00,-3.76821814e+00,  1.97972441e+00, -2.03227261e+00,1.95594284e+00,  2.69201228e+00,  7.34166875e-01,1.30401812e+00,  2.36901204e+00, -1.19630992e-03,-5.44686803e+00, -1.50312744e+00,  4.13848941e+00,7.35681087e-01,  8.19779906e+00,  8.32216723e+00,3.86105523e+00])
    test_pre,test_label = train_test_split(pica[best40_feature_list],label['REPEAT'],test_size=0.1,random_state=0)
    test_pre = np.array(test_pre)
    err = 0
    for i in range(30000,32129):
        for j in range(40):
            temp = test_pre[j][i] * intercept[j] + coef
            temp = softmax(temp)
            if temp<0.5:
                temp = 0
            else:
                temp = 1
        err = err + temp ^ test_label[i]
    ic("acc:",1-err/(32129-30000))

main()