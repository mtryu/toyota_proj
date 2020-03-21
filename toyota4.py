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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier




##############################
tool = "RandomForestClassifier Original-Test"
final =[]
for num in [0]:
    g_dic = {}
    
    for past_day in range(5,10):
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
        lena  = len(dataframe)
        
        
        a = range(0,past_day+1)
        t = range(0,past_day+2)

        for g in t:
            index = str('sub'+str(g))
            dataframe[index] = 0
        for i in range(past_day+1,lena-1):
            for  p in a:
                dataframe.iat[i,7+p] = (int(dataframe.iat[i-past_day+p+1,5])-int(dataframe.iat[i-past_day+p,5]))

            if dataframe.iat[i,7+past_day] - num > 0:
                dataframe.iat[i,8+past_day] = 1
            else:
                dataframe.iat[i,8+past_day] = 0
        #sub0 が予測したい値
        
        y_train = dataframe.iloc[past_day+1:lena-20,-1].values
        X_train = dataframe.iloc[past_day+1:lena-20, 6:7+past_day].values
        y_test = dataframe.iloc[lena-20:lena-3,-1].values
        X_test = dataframe.iloc[lena-20:lena-3, 6:7+past_day].values
        label = str('sub'+str(past_day + 1))
        count = dataframe[label].value_counts()
        act = count[1]/(count[0]+count[1])

        test_key_dic = {}
        score_trains_dic = {}
        score_tests_dic ={}
        imp_dic = {}
        predictions_dic = {}

        for p in range(past_day//2+1,past_day*2,3):
            def check_tree(max_depth, cv):
                # X_train, X_test, y_train, y_test = train_test_split(X,y, stratify = y, random_state =1)                
                # forest=RandomForestClassifier(n_estimators=p,max_depth=max_depth)
                forest = DecisionTreeClassifier(max_depth=max_depth)
                forest.fit(X_train,y_train)
                return forest, forest.predict(X_test),forest.score(X_train, y_train), forest.score(X_test, y_test),forest.feature_importances_
            #     result = cross_validate(
            #         estimator=forest, X=X, y=y, cv=cv, return_train_score=True)
            #     return forest, np.mean(result['train_score']), np.mean(result['test_score'])
            score_trains = {}
            score_tests = {}
            imp = {}
            predictions = {}
            for max_depth in range(1,30):
                _, prediction ,score_train, score_test,importance = check_tree(max_depth, cv=3)
                score_trains[max_depth] = score_train
                score_tests[max_depth] = score_test
                imp[max_depth] = importance
                predictions[max_depth] = prediction
            test_key = max(score_tests,key=score_tests.get)
            test_key_dic[p] = test_key
            score_trains_dic[p] = score_trains[test_key]
            score_tests_dic[p] = score_tests[test_key]
            imp_dic[p] = imp[test_key]
            predictions_dic[p] = predictions[test_key]
            print(str(num),"past_day:",str(past_day),"p:",str(p),"max_depth:",test_key,"train:",score_trains[test_key],"test:",score_tests[test_key],act,"imp:",str(imp[test_key]))
        f_key = max(score_tests_dic,key =score_tests_dic.get )
        dicA[past_day] = [f_key,test_key_dic[f_key],score_trains_dic[f_key],act,imp_dic[f_key],predictions_dic[f_key],y_test]
        dicB[past_day] = score_tests_dic[f_key]
        print(str(past_day),dicA,dicB)
