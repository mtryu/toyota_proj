import pandas as pd
from datetime import datetime
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

final = {}
for num in range(0,100,10):
    dataframe = pd.read_csv('7203_toyota.csv')
    dataframe["volume1"]=dataframe["volume"]
    del dataframe['volume']

    def dap(x):
        y = datetime.strptime(x,'%Y年%m月%d日')
        return y
    dataframe['date'] = dataframe['date'].apply(dap)
    def conv(ob):
        nu = int(ob.replace(',',''))
        return nu
    def conv2(ob):
        nu = int(ob.replace(',',''))
        nu = nu/100000
        return nu

    dataframe["start"] = dataframe["start"].apply(conv)
    dataframe["high"] = dataframe["high"].apply(conv)
    dataframe["low"] = dataframe["low"].apply(conv)
    dataframe["end"] = dataframe["end"].apply(conv)
    dataframe["volume1"] = dataframe["volume1"].apply(conv2)
    dataframe["final"] = dataframe["final"].apply(conv)
    dataframe["label"] = 0
    lena  = len(dataframe)
    print(dataframe)
    for i in range(1,lena-1):
        a = int(dataframe.iat[i+1,7])-int(dataframe.iat[i,7])
        if a - num >0:
            dataframe.iat[i,7] = 1
        else:
            dataframe.iat[i,7] = 0
    y = dataframe.iloc[11:-3,-1]
    X = dataframe.iloc[11:-3,-3]
    count = dataframe["label"].value_counts()
    


    score_trains={}
    score_tests={}
    test_key_dic = {}
    score_trains_dic = {}
    score_tests_dic ={}

    for p in range(1,5):
        def check_tree(X, y, max_depth):
            X_train, X_test, y_train, y_test = train_test_split(X,y, stratify = y, random_state = 42)                
            forest=RandomForestClassifier(n_estimators=p,max_depth=max_depth, random_state=2)
            forest.fit(X_train,y_train)
            return forest, forest.score(X_train, y_train), forest.score(X_test, y_test)
        for max_depth in range(1,15):
            _, score_train, score_test = check_tree(X, y, max_depth)
            score_trains[max_depth] = score_train
            score_tests[max_depth] = score_test
        test_key = max(score_tests,key=score_tests.get) #max_depth
        test_key_dic[p] = test_key
        score_trains_dic[p] = score_trains[test_key]
        score_tests_dic[p] = score_tests[test_key]
        print(str(num),"p:",str(p),"max_depth:",test_key,score_trains[test_key],score_tests[test_key])
    f_key = max(score_tests_dic,key =score_tests_dic.get )
    final[num] = [f_key,test_key_dic[f_key],score_trains_dic[f_key],score_tests_dic[f_key]]
print(final)

    

