import os
import pickle
import numpy as np


# 0. 물리 환경 및 격자 설정 (1D, L=1.0, 200개 격자점)

L = 1.0
N_GRID = 200
dx = (2.0 * L) / (N_GRID - 1)
x_grid = np.linspace(-L, L, N_GRID)

# 2차 중심차분법 기반 운동에너지 연산자 행렬 T (-1/2 * d^2/dx^2, m=1, hbar=1)
main_diag = 2.0 * np.ones(N_GRID)
off_diag = -1.0 * np.ones(N_GRID - 1)
T_matrix = (np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)) / (2.0 * (dx**2))

# 푸리에 모드 기저 (n = 1 ~ 5) 미리 생성 -> shape: (5, 200)
n_modes = np.arange(1, 6).reshape(5, 1)
k_modes = (n_modes * np.pi * x_grid[None, :]) / L
sin_basis = np.sin(k_modes)
cos_basis = np.cos(k_modes)


# print(sin_basis.shape)
#=========================================================
# 이 부분 대대적인 수정 필요하다 (위아래로) psi 복원함수 제대로 만들어야함
#=========================================================


# [단계 2] 11개 값을 입력받아서 규격화된 psi(x)를 구성하는 함수

def construct_wavefunction(params):
    """
    11차원 벡터(params)를 받아 200개 격자점 위의 규격화된 파동함수 psi(x)를 복원합니다.
    - params[0]    : 감쇠 계수 a (양수 보장 및 경계 발산 억제를 위해 softplus 변환 적용)
    - params[1:6]  : 푸리에 사인 계수 S_1 ~ S_5
    - params[6:11] : 푸리에 코사인 계수 C_1 ~ C_5
    """
    # 1. 지수 감쇠 계수 a 계산 (기본 하한선 +0.5 설정으로 경계 급격한 꺾임 방지)
    a = np.log1p(np.exp(np.clip(params[0], -20.0, 20.0))) + 0.5

    # 2. 푸리에 계수 추출
    S = params[1:6]
    C = params[6:11]

    # 3. ansatz 구성: e^(-a * x^2) * (sum S_n*sin + sum C_n*cos)
    sin_term = np.dot(S, sin_basis)
    cos_term = np.dot(C, cos_basis)
    decay = np.exp(-a * (x_grid**2))
    psi = decay * (sin_term + cos_term)

    # 4. L2 수치 규격화: integral |psi|^2 dx = 1
    norm = np.sqrt(np.sum(psi**2) * dx + 1e-12)
    psi_normalized = psi / norm

    return psi_normalized, a



# 손실함수 계산 함수 (construct_wavefunction의 출력을 사용하여 L1 + w*L2 계산)

def compute_schrodinger_loss(psi, V, w_scale):
    """
    - L1 : 슈뢰딩거 잔차 제곱합 ||H*psi - E*psi||^2 * dx
    - L2 : 바닥상태 에너지 고윳값 E = <psi|H|psi>
    - Total Loss = L1 + w * L2
    """
    # H * psi = T @ psi + V * psi
    H_psi = np.dot(T_matrix, psi) + (V * psi)

    # L2: 에너지 기대값 E
    E = np.sum(psi * H_psi) * dx

    # L1: 잔차 제곱 적분
    residual = H_psi - (E * psi)
    L1 = np.sum(residual**2) * dx

    total_loss = L1 + (w_scale * E)
    return total_loss, L1, E



# 신경망 아키텍처 및 순전파 / 역전파
# 가중치 초기화 (200 -> 128 -> 64 -> 32 -> 11)
np.random.seed(42)
weights = {
    "W1": np.random.randn(200, 128) * np.sqrt(2.0 / 200),
    "b1": np.zeros(128),
    "W2": np.random.randn(128, 64) * np.sqrt(2.0 / 128),
    "b2": np.zeros(64),
    "W3": np.random.randn(64, 32) * np.sqrt(2.0 / 64),
    "b3": np.zeros(32),
    "W4": np.random.randn(32, 11) * np.sqrt(2.0 / 32),
    "b4": np.zeros(11)
}

def forward_pass(V_batch, w):
    """퍼텐셜 V(x) 입력을 받아 11차원 파라미터 벡터를 출력"""
    a1 = np.dot(V_batch, w["W1"]) + w["b1"]
    z1 = np.tanh(a1)
    a2 = np.dot(z1, w["W2"]) + w["b2"]
    z2 = np.tanh(a2)
    a3 = np.dot(z2, w["W3"]) + w["b3"]
    z3 = np.tanh(a3)
    out = np.dot(z3, w["W4"]) + w["b4"]  # shape: (B, 11)
    cache = (V_batch, z1, z2, z3)
    return out, cache

def backward_pass(d_out, cache, w):
    """11차원 출력단 오차(d_out)를 받아 내부 층 가중치들에 대해 연쇄법칙으로 역전파"""
    V_batch, z1, z2, z3 = cache
    B = V_batch.shape[0]
    grads = {}

    # Layer 4
    grads["W4"] = np.dot(z3.T, d_out) / B
    grads["b4"] = np.sum(d_out, axis=0) / B

    # Layer 3 (tanh 미분: 1 - z^2)
    dz3 = np.dot(d_out, w["W4"].T) * (1.0 - z3**2)
    grads["W3"] = np.dot(z2.T, dz3) / B
    grads["b3"] = np.sum(dz3, axis=0) / B

    # Layer 2
    dz2 = np.dot(dz3, w["W3"].T) * (1.0 - z2**2)
    grads["W2"] = np.dot(z1.T, dz2) / B
    grads["b2"] = np.sum(dz2, axis=0) / B

    # Layer 1
    dz1 = np.dot(dz2, w["W2"].T) * (1.0 - z1**2)
    grads["W1"] = np.dot(V_batch.T, dz1) / B
    grads["b1"] = np.sum(dz1, axis=0) / B

    return grads



# [단계 1] 학습 데이터 pkl 파일 가져오기

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "train_potentials.pkl")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"데이터셋을 찾을 수 없습니다: {data_path}")

with open(data_path, "rb") as f:
    train_potentials = pickle.load(f)

print(f"[1] 학습 데이터 로드 완료: {train_potentials.shape}")



# [단계 3] 11차원 벡터 출력 -> construct_wavefunction으로 손실 계산 및 역전파

EPOCHS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 0.001
#W_SCALE = 0.01
N_SAMPLES = len(train_potentials)

# Adam Optimizer 설정
m = {k: np.zeros_like(v) for k, v in weights.items()}
v = {k: np.zeros_like(v) for k, v in weights.items()}
beta1, beta2, eps = 0.9, 0.999, 1e-8
step = 0
h_diff = 1e-5  # 11개 출력 파라미터에 대한 미소 차분값

print("\n[3] 모델 학습 시작...")
for epoch in range(1, EPOCHS + 1):
    idx_perm = np.random.permutation(N_SAMPLES)
    train_shuffled = train_potentials[idx_perm]

    loss_accum, l1_accum, l2_accum = 0.0, 0.0, 0.0
    num_batches = 0

    W_SCALE = 0 #(0.001 * (0.1 / 0.001))**((epoch - 1)/ (EPOCHS - 1))

    for i in range(0, N_SAMPLES, BATCH_SIZE):
        batch_V = train_shuffled[i:i + BATCH_SIZE]
        B = len(batch_V)
        step += 1

        # 3-1. 순전파: 퍼텐셜 입력 -> 11차원 벡터 예측
        out_params_batch, cache = forward_pass(batch_V, weights)

        # 3-2. construct_wavefunction을 통과시켜 Loss 계산 및 11개 출력단 기울기(d_out) 구하기
        d_out = np.zeros_like(out_params_batch)

        for b in range(B):
            v_sample = batch_V[b]
            p_sample = out_params_batch[b]

            # 기준 상태: 11개 값 -> psi 복원 -> 손실함수 계산
            base_psi, _ = construct_wavefunction(p_sample)
            base_loss, l1_val, l2_val = compute_schrodinger_loss(base_psi, v_sample, w_scale=W_SCALE)

            loss_accum += base_loss
            l1_accum += l1_val
            l2_accum += l2_val

            # 11개 출력 파라미터에 대한 그래디언트 수집 (dLoss / d(params_k))
            for k in range(11):
                p_shifted = p_sample.copy()
                p_shifted[k] += h_diff
                shifted_psi, _ = construct_wavefunction(p_shifted)
                shifted_loss, _, _ = compute_schrodinger_loss(shifted_psi, v_sample, w_scale=W_SCALE)
                d_out[b, k] = (shifted_loss - base_loss) / h_diff

        # 3-3. 연쇄법칙(Chain Rule) 역전파: d_out으로부터 전체 36,000개 신경망 가중치 미분값 계산
        grads = backward_pass(d_out, cache, weights)

        # 3-4. Adam 가중치 업데이트
        for k in weights:
            m[k] = beta1 * m[k] + (1.0 - beta1) * grads[k]
            v[k] = beta2 * v[k] + (1.0 - beta2) * (grads[k]**2)
            m_hat = m[k] / (1.0 - beta1**step)
            v_hat = v[k] / (1.0 - beta2**step)
            weights[k] -= LEARNING_RATE * m_hat / (np.sqrt(v_hat) + eps)

        num_batches += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch:03d}/{EPOCHS:03d}] | "
              f"Loss: {loss_accum/N_SAMPLES:.5f} | "
              f"L1 (잔차): {l1_accum/N_SAMPLES:.5f} | "
              f"L2 (에너지): {l2_accum/N_SAMPLES:.5f}")


# [단계 4] 학습 완료 후 pkl 파일 내보내기

save_path = os.path.join(current_dir, "model_weights.pkl")
with open(save_path, "wb") as f:
    pickle.dump(weights, f)

print(f"\n[4] 학습 완료! 모델 가중치가 성공적으로 저장되었습니다: {save_path}")