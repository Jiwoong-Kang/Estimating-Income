from src.data import load_adult

train = load_adult("train")
test = load_adult("test")

print(train.shape)                     # (32561, 15) 이 나와야 정상
print(test.shape)                      # (16281, 15) 이 나와야 정상
print(train["income"].unique())        # ['<=50K' '>50K'] 두 개만 나와야 정상
print(test["income"].unique())         # 테스트도 마침표 없이 두 개만 나와야 정상
print(train.isna().sum())              # workclass, occupation, native-country에만 빈 칸이 있어야 정상