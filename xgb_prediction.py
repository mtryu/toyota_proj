import pandas as pd
import xgboost as xgb

model_path = 'mod_price_prediction'

col_filter = ['area','dist_to_nearest_station','built_year']

def price_predict_df(df): #APIの中身
    # 予測モデルの読み込み
    reg = xgb.XGBRegressor()
    reg.load_model(model_path)

    return reg.predict(df)

def price_predict_q(query): #実際呼び出すAPI
    # query = [q1, q2, ...]

    data = []
    for q in query:
        data.append([float(q[col]) for col in col_filter])

    df = pd.DataFrame(data, columns=col_filter)
    df['year_pp']=2019-df['built_year']
    df=df.drop(['built_year'],axis=1)

    return price_predict_df(df)

def _test1():
    pseudo_data_path = 'sample_200_3cols.csv'
    df = pd.read_csv(pseudo_data_path) #擬似データの読み込み
    df = df.fillna(0)
    X = df[col_filter]
    
    #徒歩分をmに変換(人間が分速100㎡で進むと想定) 
    X["dist_to_nearest_station"] = X["dist_to_nearest_station"] * 100
    
    #築年から築年数を出す
    X['year_pp'] = 2019 - X['built_year']
    del X['built_year']
    
    return price_predict_df(X)

def _test2():
    q0853 = {'dist_to_nearest_station':778.63,'area':29.0,'built_year':1995} # 3238
    q1077 = {'dist_to_nearest_station':1430.52,'area':25.0,'built_year':1984} # 2852
    q2222 = {'dist_to_nearest_station':405.41,'area':29.0,'built_year':1980} # 3050
    q3844 = {'dist_to_nearest_station':596.77,'area':23.0,'built_year':1990} # 3504

    query = [q0853,q1077,q2222,q3844]
    return price_predict_q(query)

if __name__ == "__main__":
    pred = _test1()
    print(pred)
    pred2 = _test2()
    print(pred2)
    print('3238,2852,3050,3504')
