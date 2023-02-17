import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from scipy.special import comb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import svm
from itertools import combinations
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

# eps for making value a bit greater than 0 later on
eps = np.finfo(float).eps

# xxxx = np.array([
#     [0, 0, 0, 0],
#     [0, 0, 0, 1],
#     [1, 0, 0, 0],
#     [2, 1, 0, 0],
#     [2, 2, 1, 0],
#     [2, 2, 1, 1],
#     [1, 2, 1, 1],
#     [0, 1, 0, 0],
#     [0, 2, 1, 0],
#     [2, 1, 1, 0],
#     [0, 1, 1, 1],
#     [1, 1, 0, 1],
#     [1, 0, 1, 0],
#     [2, 1, 0, 1],
# ])
# yyyy = np.array([
#     0,
#     0,
#     1,
#     1,
#     1,
#     0,
#     1,
#     0,
#     1,
#     1,
#     1,
#     1,
#     1,
#     0,
# ])
#
# nn = xxxx.shape[0]
# # N = comb(nn, 2)
# xx = pd.DataFrame(xxxx)
# xx.columns = ["x1", "x2", "x3", "x4"]
# yy = pd.DataFrame(yyyy)
# yy.columns = ["y"]
# table = pd.concat([xx, yy], axis=1)


# def find_entropy(df):
#     Class = df.keys()[-1]  # To make the code generic, changing target variable class name
#     entropy = 0
#     values = df[Class].unique()
#     for value in values:
#         fraction = df[Class].value_counts()[value] / len(df[Class])
#         entropy += -fraction * np.log2(fraction)
#     return entropy
#
#
# asd = find_entropy(table)
#
#
# def find_entropy_attribute(df, attribute):
#     Class = df.keys()[-1]  # To make the code generic
#     target_variables = df[Class].unique()  # This gives all classes
#     variables = df[attribute].unique()  # This gives different features in that attribute
#     entropy2 = 0
#     for variable in variables:
#         entropy = 0
#         for target_variable in target_variables:
#             num = len(df[attribute][df[attribute] == variable][df[Class] == target_variable])
#             den = len(df[attribute][df[attribute] == variable])
#             fraction = num / (den + eps)
#             entropy += -fraction * np.log2(fraction + eps)
#         fraction2 = den / len(df)
#         entropy2 += -fraction2 * entropy
#     return abs(entropy2)
#
#
# def get_IG(df,nfet,h):
#     Entropy_att = []
#     IG = []
#     entp = find_entropy(df)
#     for key in df.keys()[:-1]:
#         # Entropy_att.append(find_entropy_attribute(df,key))
#         IG.append(entp - find_entropy_attribute(df, key))
#     combines = combinations([idx for idx in range(nfet)], h)
#
#     def sort(combine: tuple):
#         return sorted(combine, key=lambda x: IG[x], reverse=True)
#
#     # return df.keys()[:-1][np.argmax(IG)], IG
#     return IG,list(map(sort,combines))
#
#
# igggg,sorted = get_IG(table,xxxx.shape[1],2)


# classes
class FeatingEnsemble(object):
    def __init__(self, h, B, nmin):
        # var
        self.nmin = nmin
        self.tree = {}
        self.h = h
        self.B = B

    def rankAttribute(self, df, nFeat):
        def entropy(dfff):
            Class = dfff.keys()[-1]  # To make the code generic
            entropy = 0
            values = dfff[Class].unique()
            for value in values:
                fraction = dfff[Class].value_counts()[value] / len(dfff[Class])
                entropy += -fraction * np.log2(fraction)
            return entropy

        # asd = find_entropy(table)

        def entropy_attribute(dff, attribute):
            Class = dff.keys()[-1]
            target_variables = dff[Class].unique()  # This gives all classes
            variables = dff[attribute].unique()  # This gives different features in that attribute
            entropy2 = 0
            for variable in variables:
                entropy = 0
                for target_variable in target_variables:
                    num = len(dff[attribute][dff[attribute] == variable][dff[Class] == target_variable])
                    den = len(dff[attribute][dff[attribute] == variable])
                    fraction = num / (den + eps)
                    entropy += -fraction * np.log2(fraction + eps)
                fraction2 = den / len(dff)
                entropy2 += -fraction2 * entropy
            return abs(entropy2)

        # Entropy_att = []
        IG = []
        entrop = entropy(df)
        for key in df.keys()[:-1]:
            # Entropy_att.append(find_entropy_attribute(df,key))
            IG.append(entrop - entropy_attribute(df, key))
        combines = combinations([idx for idx in range(nFeat)], self.h)

        def sort(combine: tuple):
            return sorted(combine, key=lambda x: IG[x], reverse=True)

        # return df.keys()[:-1][np.argmax(IG)], IG
        return IG, list(map(sort, combines))

    def buildLevelTree(self, df, L, j):
        node = {}
        Class = df.keys()[-1]
        y = df[Class]
        x = df.drop(Class, 1)

        # Local Model
        if j == self.h:
            node['key'] = 'localModel'
            new_x = np.zeros_like(x)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    new_x[i][j] = self.tree[j][x[i, j]]
            if self.B == 'IBk':
                node['v'] = KNeighborsClassifier(n_neighbors=5).fit(new_x, y)
            elif self.B == 'J48':
                node['v'] = DecisionTreeClassifier().fit(new_x, y)
            else:
                node['v'] = SVC().fit(new_x, y)
            return node

        # return Label with majority
        if len(y) < self.nmin:
            MAX = 0
            node['key'] = 'leaf'
            for index in set(y):
                if len(y[(y == index)]) >= MAX:
                    MAX = len(y[(y == index)])
                    node['v'] = index
            return node

        if len(set(y)) == 1:
            node['key'] = 'leaf'
            node['val'] = y[0]
            return node

        node['key'] = self.rankAttribute(df, L)
        v = {}

        node['numeric'] = None
        for item in set(x[:, node['key']]):
            Xx = pd.DataFrame(x[(x[:, node['key']] == item), :])
            Yy = pd.DataFrame(y[(x[:, node['key']] == item)])
            tb = pd.concat([Xx, Yy], axis=1)
            v[item] = self.buildLevelTree(tb, np.delete(L, np.where(L == node['key']), None), j + 1)

        node['v'] = v

        return node

    def feating(self, df, h):
        Class = df.keys()[-1]
        y = df[Class]
        x = df.drop(Class, 1)

        E = list()
        n = x.shape[1]
        N = comb(n, h)
        P = self.rankAttribute(df, n)
        for i in range(N):
            L = P[i]
            E.append(self.buildLevelTree(df, L, 0))

    def fit(self, x, y):
        self.feating(x, y, self.h)

    def predict(self, df):
        return 0


# read data and train test split 70 , 30
def split_Data(path, testSplit):
    dataFrame = pd.read_csv(path)
    if path == "./segment.csv":
        DfObj = pd.DataFrame(dataFrame, list(dataFrame["class"].unique()))
        for col in dataFrame.columns:
            if DfObj.dtypes[col] == np.object:
                colData = list(dataFrame[col].unique())
                dataFrame[col] = dataFrame[col].apply(colData.index)
        Y = dataFrame['class']
        x = dataFrame.drop('class', 1)
        xTrain, xTest, yTrain, yTest = train_test_split(x, Y, test_size=testSplit, shuffle=True)

        # # Feature Scaling
        #         # sc = StandardScaler()
        #         # xTrain = sc.fit_transform(xTrain)
        #         # xTest = sc.transform(xTest)

        table = pd.concat([xTrain, yTrain], axis=1)
        return table, xTest, yTest

    elif path == "./nursery.csv":
        DfObj = pd.DataFrame(dataFrame, list(dataFrame["class"].unique()))
        for col in dataFrame.columns:
            if DfObj.dtypes[col] == np.object:
                colData = list(dataFrame[col].unique())
                dataFrame[col] = dataFrame[col].apply(colData.index)
        Y = dataFrame['class']
        x = dataFrame.drop('class', 1)
        xTrain, xTest, yTrain, yTest = train_test_split(x, Y, test_size=testSplit, shuffle=True)

        # # Feature Scaling
        # sc = StandardScaler()
        # xTrain = sc.fit_transform(xTrain)
        # xTest = sc.transform(xTest)

        table = pd.concat([xTrain, yTrain], axis=1)
        return table, xTest, yTest

    elif path == "./satimage.csv":
        Y = dataFrame['class']
        x = dataFrame.drop('class', 1)
        xTrain, xTest, yTrain, yTest = train_test_split(x, Y, test_size=testSplit, shuffle=True)

        # # Feature Scaling
        # sc = StandardScaler()
        # xTrain = sc.fit_transform(xTrain)
        # xTest = sc.transform(xTest)

        table = pd.concat([xTrain, yTrain], axis=1)
        return table, xTest, yTest

# aa,aaa,aaaa = split_Data("./segment.csv",0.3)
# cc,ccc,cccc = split_Data("./satimage.csv",0.3)
# bb,bbb,bbbb = split_Data("./nursery.csv",0.3)
