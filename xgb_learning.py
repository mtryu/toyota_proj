import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



input_path = "~/iekari_proj/download-20190826/RentRoom.csv"      #入力ファイル
output_path = "./mod_price_prediction"  #学習済みモデルファイル

def read_csv(input_path):
    data = pd.read_csv(input_path)
    return data


def preprocess(data):
    y = data[['price']]
    X = data.drop(['id', 'pref_name', 'city_name', 'district_name', 'built_year', 'structure', 'price', 'top_floor_num', 'room_type', 'nearest_station_id'], axis=1)
    X = X.drop(['latitude','longitude'], axis=1)
    X['year_pp'] = 2019 - data['built_year']

    print(X)
    
    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.3, random_state=0,
    )
    return X_train, X_test, y_train, y_test


def let_xgb_learn(X_train, y_train):
    reg = xgb.XGBRegressor()
    # ハイパーパラメータ探索
    reg_cv = GridSearchCV(reg, {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]})
    reg_cv.fit(X_train, y_train)
    print(reg_cv.best_params_, reg_cv.best_score_)

    # 改めて最適パラメータで学習
    reg = xgb.XGBRegressor(**reg_cv.best_params_)
    reg.fit(X_train, y_train)
    return reg


def check_res(reg, X_train, X_test, y_test, y_train):
    pred_train = reg.predict(X_train)
    pred_test = reg.predict(X_test)
    print(np.sqrt(mean_squared_error(y_train, pred_train)))
    print(np.sqrt(mean_squared_error(y_test, pred_test)))


def main(input_path, output_path):
    data = read_csv(input_path)
    X_train, X_test, y_train, y_test = preprocess(data)
    reg = let_xgb_learn(X_train, y_train)
    check_res(reg, X_train, X_test, y_test, y_train)
    reg.save_model(output_path)


if __name__ == "__main__":
    main(input_path, output_path)
