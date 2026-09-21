import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 0. 물리 환경 및 격자 설정 (train.py와 동일)
# ==============================================================================
L = 1.0
N_GRID = 200
dx = (2.0 * L) / (N_GRID - 1)
x_grid = np.linspace(-L, L, N_GRID)

# 푸리에 모드 기저 (n = 1 ~ 5) -> shape: (5, 200)
n_modes = np.arange(1, 6).reshape(5, 1)
k_modes = (n_modes * np.pi * x_grid[None, :]) / L
sin_basis = np.sin(k_modes)
cos_basis = np.cos(k_modes)

# 운동에너지 연산자 행렬 T (2차 중심차분법, m=1, hbar=1)
main_diag = 2.0 * np.ones(N_GRID)
off_diag = -1.0 * np.ones(N_GRID - 1)
T_matrix = (np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)) / (2.0 * (dx**2))


# ==============================================================================
# 1. 11개 값을 입력받아 psi(x)를 구성하는 함수 (train.py와 완전히 동일)
# ==============================================================================
def construct_wavefunction(params):
    """
    11차원 벡터(params)를 받아 규격화된 파동함수 psi(x)를 복원합니다.
    - params[0]    : 감쇠 계수 a
    - params[1:6]  : 푸리에 사인 계수 S_1 ~ S_5
    - params[6:11] : 푸리에 코사인 계수 C_1 ~ C_5
    """
    a = np.log1p(np.exp(np.clip(params[0], -20.0, 20.0))) + 0.5
    S = params[1:6]
    C = params[6:11]

    sin_term = np.dot(S, sin_basis)
    cos_term = np.dot(C, cos_basis)
    decay = np.exp(-a * (x_grid**2))

    psi = decay * (sin_term + cos_term)

    # 수치 규격화: integral |psi|^2 dx = 1
    norm = np.sqrt(np.sum(psi**2) * dx + 1e-12)
    psi_normalized = psi / norm

    # 시각화 비교 편의를 위해 중심부 피크 부호를 항상 양수로 통일
    if psi_normalized[N_GRID // 2] < 0:
        psi_normalized = -psi_normalized

    return psi_normalized, a


# ==============================================================================
# 2. 저장된 가중치(model_weights.pkl) 로드 및 순전파 함수 정의
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
weight_path = os.path.join(current_dir, "model_weights.pkl")

if not os.path.exists(weight_path):
    raise FileNotFoundError(f"가중치 파일이 없습니다. 먼저 train.py를 실행하세요: {weight_path}")

with open(weight_path, "rb") as f:
    weights = pickle.load(f)

def forward_network(V_input, w):
    """(200,) 크기의 퍼텐셜 벡터를 입력받아 11차원 파라미터 벡터를 출력"""
    z1 = np.tanh(np.dot(V_input, w["W1"]) + w["b1"])
    z2 = np.tanh(np.dot(z1, w["W2"]) + w["b2"])
    z3 = np.tanh(np.dot(z2, w["W3"]) + w["b3"])
    out = np.dot(z3, w["W4"]) + w["b4"]  # shape: (11,)
    return out


# ==============================================================================
# 3. 테스트할 퍼텐셜 준비 (예시: 조화진동자 k=50)
# ==============================================================================
k_spring = 50.0
omega = np.sqrt(k_spring)
V_test = 0.5 * k_spring * (x_grid**2)  # 테스트용 V(x)

# 이론적 해석해 (조화진동자 바닥상태 가우시안 및 에너지)
exact_psi = (omega / np.pi)**0.25 * np.exp(-0.5 * omega * (x_grid**2))
exact_psi /= np.sqrt(np.sum(exact_psi**2) * dx)
exact_energy = 0.5 * omega


# ==============================================================================
# 4. 모델 추론: V(x) -> 11차원 파라미터 -> construct_wavefunction -> psi(x)
# ==============================================================================
pred_params = forward_network(V_test, weights)
pred_psi, pred_a = construct_wavefunction(pred_params)

# 슈뢰딩거 잔차 및 에너지 고윳값 수치 평가
H_psi = np.dot(T_matrix, pred_psi) + (V_test * pred_psi)
pred_energy = np.sum(pred_psi * H_psi) * dx
residual = H_psi - (pred_energy * pred_psi)
residual_loss = np.sum(residual**2) * dx


# ==============================================================================
# 5. 콘솔 평가 지표 출력
# ==============================================================================
print("=" * 55)
print("             [모델 테스트 평가 결과]")
print("=" * 55)
print(f"예측된 11개 파라미터 :")
print(f"  - 감쇠 계수 a      : {pred_a:.4f} (이론값 m*w/2 = {0.5*omega:.4f})")
print(f"  - 사인 계수 S[1:5] : {np.round(pred_params[1:6], 4)}")
print(f"  - 코사인 계수 C[1:5]: {np.round(pred_params[6:11], 4)}")
print("-" * 55)
print(f"이론 바닥상태 에너지 (Exact E)  : {exact_energy:.4f}")
print(f"신경망 예측 에너지   (Pred E)   : {pred_energy:.4f}")
print(f"에너지 오차율                   : {abs(pred_energy - exact_energy)/exact_energy * 100:.2f} %")
print(f"슈뢰딩거 잔차 L1 (||Hψ - Eψ||²) : {residual_loss:.5f}")
print("=" * 55)


# ==============================================================================
# 6. 시각화 (Matplotlib)
# ==============================================================================
plt.figure(figsize=(10, 6))

# 이론해 vs 신경망 예측 파동함수
plt.plot(x_grid, exact_psi, 'k--', linewidth=2, label=f"Exact $\psi_0(x)$ (E={exact_energy:.3f})")
plt.plot(x_grid, pred_psi, 'b-', linewidth=2, label=f"NN Pred $\psi(x)$ (E={pred_energy:.3f})")
plt.fill_between(x_grid, exact_psi, pred_psi, color='red', alpha=0.2, label="Difference")

# 퍼텐셜 형태 시각 가이드 (스케일 조정)
V_scaled = (V_test / np.max(V_test)) * np.max(exact_psi)
plt.plot(x_grid, V_scaled, 'gray', linestyle=':', alpha=0.5, label="V(x) shape (scaled)")

plt.axhline(y=0.0, color='black', linewidth=0.8, alpha=0.3)
plt.title("Comparison: Theoretical vs NN Harmonic Oscillator Ground State", fontsize=13)
plt.xlabel("Position x", fontsize=11)
plt.ylabel("$\psi(x)$", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.show()