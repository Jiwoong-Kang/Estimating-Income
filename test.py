from src.data import load_adult
from src.preprocess import split_features_target, _categorical_steps, CATEGORICAL_COLS, CAPITAL_COLS, _capital_steps, build_preprocessor, make_model_pipeline
from sklearn.linear_model import LogisticRegression
X, y = split_features_target(load_adult("train"))

# print(X.shape)
# print(list(X.columns))
# print(y.value_counts())

# cat = _categorical_steps()
# Z = cat.fit_transform(X[CATEGORICAL_COLS])
# print(Z.shape)
# print(cat.get_feature_names_out()[:5])

# cap = _capital_steps()
# C = cap.fit_transform(X[CAPITAL_COLS])
# print(C.shape)                      # (32561, 2) 가 나와야 함
# print(C[:, 0].min(), C[:, 0].max()) # 원래 0~99999 → 대략 -0.3 ~ 4.5 정도로 줄어듦

# pre = build_preprocessor()
# P = pre.fit_transform(X)            # X 전체를 넣으면 알아서 열을 나눠 손질한다
# print(P.shape)                      # (32561, 71) 이 나와야 함
# print(pre.get_feature_names_out()[:6])

test_X, test_y = split_features_target(load_adult("test"))
pipe = make_model_pipeline(LogisticRegression(max_iter=1000))
pipe.fit(X, y)                       # 훈련 데이터로 전처리 배우기 + 모델 학습
print(pipe.score(test_X, test_y))    # 시험 데이터 정확도