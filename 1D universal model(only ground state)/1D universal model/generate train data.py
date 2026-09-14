import pickle
import numpy as np
import os

# 1. 단일 퍼텐셜 계산 함수 (x가 200개 배열이면 V도 200개 배열로 즉시 계산)
def generate_1d_data(x, V0, x0, w, s):
    arg = (np.abs(x - x0) - w / 2.0) / s
    arg = np.clip(arg, -60.0, 60.0)  # overflow 방지
    V = -V0 * (1.0 + np.exp(arg))**-1
    return V

# 2. 격자점 생성 및 퍼텐셜 벡터 반환
def generate_train_data(L, V0, x0, w, s):
    train_data_x = np.linspace(-L, L, 200)  # 모델 입력 크기 200에 맞춤
    train_data_V = generate_1d_data(train_data_x, V0, x0, w, s)  # 루프 없이 한 번에 계산
    return train_data_x, train_data_V

# 3. 데이터셋 파라미터 무작위 샘플링
data_size = 1000
L = 1.0
V0 = np.random.uniform(5.0, 50.0, size=data_size)
x0 = 0.0
w  = np.random.uniform(0.0, 0.8, size=data_size)
s  = np.random.uniform(0.02, 0.25, size=data_size)

# (1000, 200) 크기의 2차원 배열 공간 미리 할당
F_train_data_V = np.zeros((data_size, 200))

# 4. 데이터 생성 루프
for i in range(data_size):
    # i번째 파라미터 전달 (i-1 대신 i 사용)
    _, V_vec = generate_train_data(L, V0[i], x0, w[i], s[i])
    F_train_data_V[i] = V_vec  # i번째 행에 채워넣기

print("생성 완료! Shape:", F_train_data_V.shape)  # (1000, 200) 출력

# 5. 실행 중인 파이썬 스크립트 파일이 위치한 디렉터리 경로를 구함
current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, "train_potentials.pkl")

# 해당 폴더 경로에 저장
with open(save_path, "wb") as f:
    pickle.dump(F_train_data_V, f)

print(f"저장 완료: {save_path}")





"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 1차원 모델의 학습 데이터 생성 함수
# 우물 형태의 데이터들로 구성
def generate_1d_data (x, V0, x0, w, s):
    V = - V0 * (1 + np.exp((abs(x - x0)-w/2)/s))**-1

    return V

def generate_train_data (L, V0, x0, w, s):
    train_data_x = np.linspace(-L,L,100)
    train_data_V = np.zeros_like(train_data_x)

    for i in range (100) :
        train_data_V[i-1] = generate_1d_data(train_data_x[i-1], V0, x0, w, s)

    return train_data_x, train_data_V

#데이터 생성 루프
data_size = 1000
L = 1.0
V0 = np.random.uniform(5.0,50.0,size = data_size)
x0 = 0
w = np.random.uniform(0.0,0.8,size = data_size)
s = np.random.uniform(0.02,0.25,size = data_size)

train_data = []

for i in range(data_size) :
    N_train_data = generate_train_data(L,V0[i-1],x0,w[i-1],s[i-1])
    F_train_data_V[i-1] = N_train_data[1]


print(F_train_data_V)

"""