import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

data_path = '/Users/sola/Downloads/open/'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')

train_df = train.drop(['PRODUCT_ID','PRODUCT_CODE','LINE','Y_Class','Y_Quality','TIMESTAMP'], axis=1)
test_df = test.drop(['TIMESTAMP'], axis=1)

train_df= train_df.fillna(0)
test_df= test_df.fillna(0)

X_train = train_df
y = train['Y_Class']

#검증데이터..
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y, test_size=0.3, stratify=y, random_state=37)

hgb = HistGradientBoostingClassifier(random_state=37)
scores = cross_validate(hgb, X_train, y_train, return_train_score=True, n_jobs=-1)
# print(np.mean(scores['train_score']), np.mean(scores['test_score']))
#0.9321723946453317 0.8801241948619236

hgb.fit(X_train, y_train)
#permutation_importance는 특성을 하나씩 랜덤하게 섞어서 모델의 성능이 변화하는지를 관찰하여 어떤 특성이 중요한지 계산한다.
#n_repeats 매개변수는 랜덤하게 섞을 횟수를 지정한다.
result = permutation_importance(hgb, X_train, y_train, n_repeats=5, random_state=37, n_jobs=-1)
print(result.importances_mean)

#[0.08876275 0.23438522 0.08027708]


# result = permutation_importance(hgb, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1)
# print(result.importances_mean)
#[0.05969231 0.20238462 0.049     ]


# hgb.score(X_train, y_train)
#0.8723076923076923