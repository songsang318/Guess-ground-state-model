import os
import pickle
import numpy as np

# 1. 물리 파라미터 및 격자 설정
L = 1.0
N_GRID = 200
dx = (2.0 * L) / (N_GRID - 1)
x_grid = np.linspace(-L, L, N_GRID)

# T matrix (2차 중심차분)
main_diag = 2.0 * np.ones(N_GRID)
off_diag = -1.0 * np.ones(N_GRID - 1)
T_mat = (np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)) / (2.0 * (dx**2))

# Fourier 기저 행렬: (5, 200)
n_modes = np.arange(1, 6).reshape(5, 1)
k_modes = (n_modes * np.pi * x_grid[None, :]) / L
sin_basis = np.sin(k_modes)
cos_basis = np.cos(k_modes)

# 2. 신경망 초기화
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

# 3. 모델 순전파 (캐싱 포함)
def forward_pass(V_batch, w):
    # z: 활성화 후, a: 선형 결합
    a1 = np.dot(V_batch, w["W1"]) + w["b1"]
    z1 = np.tanh(a1)
    a2 = np.dot(z1, w["W2"]) + w["b2"]
    z2 = np.tanh(a2)
    a3 = np.dot(z2, w["W3"]) + w["b3"]
    z3 = np.tanh(a3)
    out = np.dot(z3, w["W4"]) + w["b4"] # (B, 11)
    cache = (V_batch, a1, z1, a2, z2, a3, z3, out)
    return out, cache

# 4. 단일 샘플의 Loss 및 dLoss/d(out_params) 수치 그래디언트 (11개만 미분하므로 즉시 연산됨)
def get_sample_loss_and_grad(params, V, w_scale=0.01):
    def loss_func(p):
        a = np.log1p(np.exp(np.clip(p[0], -20.0, 20.0))) + 0.1
        S, C = p[1:6], p[6:11]
        psi = np.exp(-a * (x_grid**2)) * (np.dot(S, sin_basis) + np.dot(C, cos_basis))
        norm = np.sqrt(np.sum(psi**2) * dx + 1e-12)
        psi_n = psi / norm
        H_psi = np.dot(T_mat, psi_n) + V * psi_n
        E = np.sum(psi_n * H_psi) * dx
        res = H_psi - E * psi_n
        L1 = np.sum(res**2) * dx
        return L1 + w_scale * E, L1, E

    base_loss, l1, l2 = loss_func(params)
    
    # 36,000개가 아닌 출력 파라미터 11개에 대해서만 미분
    h = 1e-5
    d_params = np.zeros_like(params)
    for i in range(11):
        p_plus = params.copy()
        p_plus[i] += h
        l_plus, _, _ = loss_func(p_plus)
        d_params[i] = (l_plus - base_loss) / h

    return base_loss, l1, l2, d_params

# 5. 역전파 (Backpropagation: 출력층부터 입력층까지 연쇄법칙)
def backward_pass(d_out, cache, w):
    V_batch, a1, z1, a2, z2, a3, z3, out = cache
    B = V_batch.shape[0]

    grads = {}
    grads["W4"] = np.dot(z3.T, d_out) / B
    grads["b4"] = np.sum(d_out, axis=0) / B

    dz3 = np.dot(d_out, w["W4"].T) * (1.0 - z3**2)
    grads["W3"] = np.dot(z2.T, dz3) / B
    grads["b3"] = np.sum(dz3, axis=0) / B

    dz2 = np.dot(dz3, w["W3"].T) * (1.0 - z2**2)
    grads["W2"] = np.dot(z1.T, dz2) / B
    grads["b2"] = np.sum(dz2, axis=0) / B

    dz1 = np.dot(dz2, w["W2"].T) * (1.0 - z1**2)
    grads["W1"] = np.dot(V_batch.T, dz1) / B
    grads["b1"] = np.sum(dz1, axis=0) / B

    return grads

# 6. Adam Optimizer 변수
m = {k: np.zeros_like(v) for k, v in weights.items()}
v = {k: np.zeros_like(v) for k, v in weights.items()}
lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-8

# 7. 데이터 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "train_potentials.pkl")
with open(data_path, "rb") as f:
    train_potentials = pickle.load(f)

print(f"데이터셋 로드 완료: {train_potentials.shape}")

# 8. 최적화된 학습 루프
EPOCHS = 1000
BATCH_SIZE = 32
N_SAMPLES = len(train_potentials)
t = 0

for epoch in range(1, EPOCHS + 1):
    idx_perm = np.random.permutation(N_SAMPLES)
    train_shuffled = train_potentials[idx_perm]

    loss_accum, l1_accum, l2_accum = 0.0, 0.0, 0.0
    num_batches = 0

    for i in range(0, N_SAMPLES, BATCH_SIZE):
        batch_V = train_shuffled[i:i + BATCH_SIZE]
        B = len(batch_V)
        t += 1

        out_params, cache = forward_pass(batch_V, weights)

        # 배치 내 샘플별로 11차원 벡터 미분 수집
        d_out = np.zeros_like(out_params)
        for b in range(B):
            l, l1, l2, dp = get_sample_loss_and_grad(out_params[b], batch_V[b])
            loss_accum += l
            l1_accum += l1
            l2_accum += l2
            d_out[b] = dp

        # 고속 역전파 행렬곱 계산
        grads = backward_pass(d_out, cache, weights)

        # Adam 갱신
        for k in weights:
            m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
            v[k] = beta2 * v[k] + (1 - beta2) * (grads[k]**2)
            m_hat = m[k] / (1 - beta1**t)
            v_hat = v[k] / (1 - beta2**t)
            weights[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        num_batches += 1

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch [{epoch:03d}/{EPOCHS:03d}] | "
              f"Loss: {loss_accum/N_SAMPLES:.5f} | "
              f"L1 (Res): {l1_accum/N_SAMPLES:.5f} | "
              f"L2 (Energy): {l2_accum/N_SAMPLES:.5f}")

# 9. 모델 저장
save_path = os.path.join(current_dir, "model_weights.pkl")
with open(save_path, "wb") as f:
    pickle.dump(weights, f)

print(f"학습 완료 및 저장: {save_path}")