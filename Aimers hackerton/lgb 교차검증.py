import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import RidgeClassifier
import lightgbm as lgb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from warnings import simplefilter

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold



import matplotlib.pyplot as plt
import seaborn as sns


simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

lines = ['T050304', 'T050307', 'T100304', 'T100306', 'T010306', 'T010305']
def corr_matrix():
    df = pd.read_csv("Dataset/train.csv")
    # print(df.corr())
    df=df.fillna(0)
    # print(df.head())
    df = df.corr()
    print(np.isfinite(df))
    # sns.clustermap(df,
    #                annot=True,  # 실제 값 화면에 나타내기
    #                cmap='RdYlBu_r',  # Red, Yellow, Blue 색상으로 표시
    #                vmin=-1, vmax=1,  # 컬러차트 -1 ~ 1 범위로 표시
    #                )

def seperate_code():
    output_dir = "Input/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv("Dataset/train.csv")

    for line in lines:
        df_line = df[df["LINE"] == line]
        df_line = df_line.dropna(how="any", axis="columns")

        # 모든 값이 동일한 열 제거
        cols = df_line.columns[6:]
        for col in cols:
            if len(df_line[col].unique()) == 1:
                df_line = df_line.drop([col], axis=1)

        df_line.to_csv(os.path.join(output_dir, f"{line}.csv"), index=False)


def find_duplicate_col():
    df = pd.read_csv("Dataset/train.csv")
    df = df.round(0)

    drop_cols = []
    for line in lines:
        df_line = df[df["LINE"] == line]
        df_line = df_line.dropna(how="any", axis="columns")

        df_line = df_line.loc[:, ~df.T.duplicated()]

        cols = df_line.columns[6:]
        for col in cols:
            if len(df_line[col].unique()) == 1:
                df_line = df_line.drop([col], axis=1)

        drop_cols.append(df_line.columns)

    return drop_cols


def make_col():
    col_list = []
    for l, line in enumerate(lines):
        df = pd.read_csv(f"Input/{line}.csv")

        X = df[find_duplicate_col()[l]].drop(["Y_Class", "Y_Quality"], axis=1).select_dtypes(exclude=['object'])

        X.to_csv(f"Input/{line}.csv", index=False)
        col_list.append(X.columns)

    # Code : A
    A_col = list(set(col_list[0]) & set(col_list[1]) & set(col_list[4]) & set(col_list[5]))
    # Code : T, O
    OT_col = list(set(col_list[2]) & set(col_list[3]))

    use_col = A_col + OT_col

    return use_col, A_col, OT_col


def data_modeling_A():
    df = pd.read_csv("Dataset/train.csv")

    data_A = df[df["PRODUCT_CODE"] == "A_31"]

    A_y = data_A["Y_Class"]
    A_X = data_A.drop(["Y_Class", "Y_Quality"], axis=1).select_dtypes(exclude=['object'])

    X_train, X_test, y_train, y_test = train_test_split(A_X, A_y, test_size=0.3, random_state=42)

    df = A_X.corr()
    print(np.isfinite(df))

    model_A = lgb.LGBMClassifier(random_state=42)
    model_A.fit(X_train, y_train)
    prediction = model_A.predict(X_test)

    print(confusion_matrix(prediction, y_test))
    print(f"A_정답률:{accuracy_score(prediction, y_test):.3f}")

    scores = cross_validate(model_A, X_train, y_train, cv=StratifiedKFold())
    print(np.mean(scores['test_score']))

    # n_splits매개변수는 몇(k)폴드 교차 검증 할지를 정한다.
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_validate(model_A, X_train, y_train, cv=splitter)
    print(np.mean(scores['test_score']))

    return model_A, A_X.columns

def model_T_gridSearch():
    df = pd.read_csv("Dataset/train.csv")
    data_T = df[(df["PRODUCT_CODE"] == "T_31") | (df["PRODUCT_CODE"] == "O_31")]

    lists = []
    for a in make_col()[2]:
        lists.append(int(a[2:]))
    lists.sort()

    cols = []  # 사용할 columns
    for l in lists:
        cols.append(f"X_{l}")

    T_y = data_T["Y_Class"]
    T_X = data_T.drop("Y_Class", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(T_X, T_y, test_size=0.3, random_state=42)

    # 그리드서치(하이퍼파라미터튜닝ing)
    model_T = RidgeClassifier(random_state=42)
    ridge_params = {'max_iter': [3000], 'alpha': [0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300, 400, 800, 900, 1000]}
    # rmsle_score = metrics.make_scorer(rmsle, greater_is_better=False)

    gs = GridSearchCV(estimator=model_T,
                      param_grid=ridge_params,
                      n_jobs=-1,
                      error_score='raise')

    gs.fit(X_train, y_train)

    # 교차검증에서 최적의 하이퍼파라미터를 찾으면 전체 훈련 세트로 모델을 다시 만들어야한다.
    # -> 그리드서치는 best_estimator_을 통해 자동으로 다시 모델을 훈련할 수 있다.
    dt = gs.best_estimator_
    print(dt.score(X_train, y_train))

    # 그리드 서치에서 찾은 최적의 매개변수는 best_params에 저장되어있다.
    print(gs.best_params_)

    # 각 매개변수에서 수행한 교차 검증의 평균 점수는 mean_test_score에 저장되어있다.
    print(gs.cv_results_['mean_test_score'])

    # 넘파이의 argmax()를 통해 가장 큰 값의 인덱스를 추출하고
    # 이 인덱스를 사용하여 params 키에 저장된 매개변수를 출력할 수 있다.
    best_index = np.argmax(gs.cv_results_['mean_test_score'])
    print(gs.cv_results_['params'][best_index])

def data_modeling_T():
    df = pd.read_csv("Dataset/train.csv")
    data_T = df[(df["PRODUCT_CODE"] == "T_31") | (df["PRODUCT_CODE"] == "O_31")]

    lists = []
    for a in make_col()[2]:
        lists.append(int(a[2:]))
    lists.sort()

    cols = [] # 사용할 columns
    for l in lists:
        cols.append(f"X_{l}")

    T_y = data_T["Y_Class"]
    T_X = data_T[cols]

    X_train, X_test, y_train, y_test = train_test_split(T_X, T_y, test_size=0.3, random_state=42)
    model_T = RidgeClassifier(random_state=42)
    model_T.fit(X_train, y_train)
    prediction = model_T.predict(X_test)


    #혼동행렬과 정답률
    print(confusion_matrix(prediction, y_test))
    print(f"T_정답률:{accuracy_score(prediction, y_test):.3f}")


    # 교차검증!!
    scores = cross_validate(model_T, X_train, y_train, cv=StratifiedKFold())
    print(np.mean(scores['test_score']))
    # n_splits매개변수는 몇(k)폴드 교차 검증 할지를 정한다.
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_validate(model_T, X_train, y_train, cv=splitter)
    print(np.mean(scores['test_score']))

    return model_T, cols


# testset에 전처리 및 제출
def predict_testset():
    df = pd.read_csv("Dataset/test.csv")

    data_A = data_modeling_A()
    data_T = data_modeling_T()

    # A_testset
    df_A = df[df["PRODUCT_CODE"] == "A_31"]

    df_test_A = pd.DataFrame()
    A_cols = data_A[1]

    for c, col in enumerate(A_cols):
        df_test_A[col] = df_A[col]

    # T_testset
    df_T = df[(df["PRODUCT_CODE"] == "T_31") | (df["PRODUCT_CODE"] == "O_31")]

    df_test_T = pd.DataFrame()
    T_cols = data_T[1]

    for c, col in enumerate(T_cols):
        df_test_T[col] = df_T[col]

    pred_A = data_A[0].predict(df_test_A)
    pred_T = data_T[0].predict(df_test_T)

    # Code A, T 순서 원래대로 변경
    A_index = df[df["PRODUCT_CODE"] == "A_31"].index
    T_index = df[(df["PRODUCT_CODE"] == "T_31") | (df["PRODUCT_CODE"] == "O_31")].index

    A_add_df = pd.DataFrame()
    A_add_df["index"] = A_index
    A_add_df["predict"] = pred_A

    T_add_df = pd.DataFrame()
    T_add_df["index"] = T_index
    T_add_df["predict"] = pred_T

    predict_data = pd.concat([A_add_df, T_add_df])
    predict_data = predict_data.sort_values(by=["index"], ascending=[True])

    # submit
    output_dir = "Result/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    submit = pd.read_csv("Dataset/sample_submission.csv")
    submit["Y_Class"] = predict_data["predict"].values
    submit.to_csv(os.path.join(output_dir, "./cross_val_lgb_submission.csv"), index=False)



def main():
    seperate_code()
    # corr_matrix()
    # seperate_code()
    # # predict_testset()
    # # predict_testset()
    # data_modeling_A()
    # data_modeling_T()
    model_T_gridSearch()



if __name__ == "__main__":
    main()