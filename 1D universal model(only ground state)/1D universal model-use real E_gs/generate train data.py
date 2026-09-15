import pickle
import numpy as np
import os

# 1차원 모델이 학습할 퍼텐셜 함수
# 우물 형태이며 여러 변수 조작을 통해 넓이, 깊이, 뭉툭한 정도 등을 조절가능
def generate_1d_data(x, V0, x0, w, s):
    V = -V0 * (1.0 + np.exp((np.abs(x - x0) - w / 2.0) / s))**-1
    return V

# 퍼텐셜 함수를 데이터 세트로 분해하는 함수
def generate_train_data(L, V0, x0, w, s):
    train_data_x = np.linspace(-L, L, 200)  
    train_data_V = generate_1d_data(train_data_x, V0, x0, w, s)  
    return train_data_x, train_data_V

# 데이터셋 파라미터 무작위 샘플링
data_size = 1000 # 학습 데이터(퍼텐셜 함수)의 개수
L = 1.0
V0 = np.random.uniform(5.0, 50.0, size=data_size)
x0 = 0.0
w  = np.random.uniform(0.0, 0.8, size=data_size)
s  = np.random.uniform(0.02, 0.25, size=data_size)

# final train data, 미리 빈 칸을 만들어놓기 (append 써도 되는데 헷갈림)
F_train_data_V = np.zeros((data_size, 200))

# 데이터 생성 루프
for i in range(data_size):
    # generate train data 함수에서 train data x는 필요 없으니 train data V만 받음
    _, V_vec = generate_train_data(L, V0[i], x0, w[i], s[i])
    F_train_data_V[i] = V_vec  # i번째 행에 채워넣기

print("생성 완료! Shape:", F_train_data_V.shape)  # (1000, 200) 출력

# 실행 중인 파이썬 스크립트 파일이 위치한 디렉터리 경로를 구함
current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, "train_potentials.pkl")

# 해당 폴더 경로에 저장
with open(save_path, "wb") as f:
    pickle.dump(F_train_data_V, f)

print(f"저장 완료: {save_path}")