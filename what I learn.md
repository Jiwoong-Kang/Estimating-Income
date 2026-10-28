# What I Learn

프로젝트를 진행하면서 배운 점을 기록하는 문서. 나중에 리포트의 "Lessons Learnt" 부분을 쓸 때 참고한다.
자세한 내용은 `learn/` 폴더에 날짜별로 나눠 기록한다.

| 날짜 | 주제 | 핵심 |
|---|---|---|
| [2026-10-12](learn/2026-10-12.md) | 프로젝트 준비 | 프로젝트 방향, 데이터셋 특징, 가상환경, `src/data.py`, 오류 없이 틀리는 실수 |
| [2026-10-13](learn/2026-10-13.md) | EDA (탐색적 데이터 분석) | 빈 칸의 의미, 정확도의 함정, 분포와 상한값, 상관관계 지표, 분석은 결정으로 끝나야 한다 |
| [2026-10-14](learn/2026-10-14.md) | 전처리 모듈 (`src/preprocess.py`) | 원-핫 인코딩, 로그 변환과 크기 맞추기, Pipeline과 ColumnTransformer, fit/transform과 데이터 누수 |
| [2026-10-16](learn/2026-10-16.md) | 평가 모듈 + Dummy 기준선 | StratifiedKFold, 여러 지표, 혼동 행렬, results json, Dummy(most_frequent) 바닥선 |
| [2026-10-17](learn/2026-10-17.md) | 선형·다항 회귀 | 임계값 래퍼, 안쪽 Pipeline(다항→선형), deg=2 결과, import/커널 |
| [2026-10-19](learn/2026-10-19.md) | Perceptron | decision_function, random_state, Dummy보다 낫지만 불안정·회귀보다 약함 |
| [2026-10-20](learn/2026-10-20.md) | SVM ~ Naive Bayes | LinearSVC, k-NN, Tree, NB(dense, 정확도가 Dummy보다 낮아도 정상) |
| [2026-10-21](learn/2026-10-21.md) | Logistic Regression | 확률 모델, SVM·선형 회귀와 비슷한 실력, 비교용 추가 |
| [2026-10-23](learn/2026-10-23.md) | 하이퍼파라미터 튜닝 (Poly~kNN) | deg=2, Perceptron `alpha`, SVM `C=1`, kNN `k=35` |
| [2026-10-24](learn/2026-10-24.md) | Tree / NB / Logistic 튜닝 | Tree depth=15, NB smoothing=0.1, Logistic `C=10`(민감도 낮음) |
| [2026-10-26](learn/2026-10-26.md) | K-Means 클러스터링 | Elbow+Silhouette, k=2, 클러스터별 고소득 비율(20.6% vs 62%) |
| [2026-10-28](learn/2026-10-28.md) | Bias–Variance | train vs CV로 간접 설명, Tree/kNN/Poly, |