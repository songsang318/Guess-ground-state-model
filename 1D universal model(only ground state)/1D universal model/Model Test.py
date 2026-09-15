import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 1. 격자 및 물리 환경 설정 (train.py와 동일)
L = 1.0
N_GRID = 200
dx = (2.0 * L) / (N_GRID - 1)
x_grid = np.linspace(-L, L, N_GRID)

# Fourier 기저 행렬 (5, 200)
n_modes = np.arange(1, 6).reshape(5, 1)
k_modes = (n_modes * np.pi * x_grid[None, :]) / L
sin_basis = np.sin(k_modes)
cos_basis = np.cos(k_modes)

# 2. 테스트용 조화진동자 퍼텐셜 생성: V(x) = 1/2 * k * x^2
# 양 끝(x = ±1)에서 파동함수가 0으로 잘 감쇠하도록 k 설정 (예: k = 50 -> omega = sqrt(50) ≈ 7.07)
k_spring = 50.0
omega = np.sqrt(k_spring)
V_harmonic = 0.5 * k_spring * (x_grid**2)  # shape: (200,)

# 3. 조화진동자 이론적 해석해 (Ground State: psi_0(x) = (m*w / pi*hbar)^(1/4) * exp(- m*w*x^2 / 2*hbar))
# m=1, hbar=1 단위계
exact_psi = (omega / np.pi)**0.25 * np.exp(-0.5 * omega * (x_grid**2))
exact_energy = 0.5 * omega  # E_0 = 1/2 * hbar * omega

# 수치 규격화 확인
exact_psi /= np.sqrt(np.sum(exact_psi**2) * dx)

# 4. 저장된 신경망 가중치 불러오기
current_dir = os.path.dirname(os.path.abspath(__file__))
weight_path = os.path.join(current_dir, "model_weights.pkl")

if not os.path.exists(weight_path):
    raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weight_path}")

with open(weight_path, "rb") as f:
    weights = pickle.load(f)

# 5. 신경망 순전파 (Forward Pass)
def forward_pass(V_input, w):
    z1 = np.tanh(np.dot(V_input, w["W1"]) + w["b1"])
    z2 = np.tanh(np.dot(z1, w["W2"]) + w["b2"])
    z3 = np.tanh(np.dot(z2, w["W3"]) + w["b3"])
    out = np.dot(z3, w["W4"]) + w["b4"]
    return out

# 6. 출력 파라미터(11개) -> 파동함수 복원
def construct_wavefunction(params):
    a = np.log1p(np.exp(np.clip(params[0], -20.0, 20.0))) + 0.1
    S = params[1:6]
    C = params[6:11]

    sin_term = np.dot(S, sin_basis)
    cos_term = np.dot(C, cos_basis)
    decay = np.exp(-a * (x_grid**2))

    psi = decay * (sin_term + cos_term)
    
    # 규격화
    norm = np.sqrt(np.sum(psi**2) * dx + 1e-12)
    psi = psi / norm

    # 부호 보정 (피크 방향을 이론해와 일치시키기 위해 양수로 통일)
    if psi[N_GRID // 2] < 0:
        psi = -psi
    return psi, a

# 신경망 예측 실행
pred_params = forward_pass(V_harmonic, weights)
pred_psi, pred_a = construct_wavefunction(pred_params)

# 7. 예측된 파동함수의 에너지 기댓값 계산
# T 연산자 행렬 생성
main_diag = 2.0 * np.ones(N_GRID)
off_diag = -1.0 * np.ones(N_GRID - 1)
T_mat = (np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)) / (2.0 * (dx**2))

H_pred_psi = np.dot(T_mat, pred_psi) + (V_harmonic * pred_psi)
pred_energy = np.sum(pred_psi * H_pred_psi) * dx

# 슈뢰딩거 잔차 계산
res = H_pred_psi - pred_energy * pred_psi
res_error = np.sum(res**2) * dx

# 8. 콘솔 결과 출력
print("=" * 45)
print(f"조화진동자 특성각진동수 (omega): {omega:.3f}")
print(f"이론적 바닥상태 에너지 (E_exact) : {exact_energy:.4f}")
print(f"신경망 예측 에너지     (E_pred)  : {pred_energy:.4f}")
print(f"에너지 오차율                   : {abs(pred_energy - exact_energy) / exact_energy * 100:.2f}%")
print(f"슈뢰딩거 잔차 (||Hpsi - Epsi||^2): {res_error:.5f}")
print(f"예측된 감쇠 계수 a               : {pred_a:.4f} (이론치 m*w/2: {0.5*omega:.4f})")
print("=" * 45)

# 9. 시각화 (Matplotlib)
plt.figure(figsize=(10, 6))

# 파동함수 비교 플롯
plt.plot(x_grid, exact_psi, 'k--', linewidth=2, label=f"Exact $\psi_0(x)$ (E={exact_energy:.3f})")
plt.plot(x_grid, pred_psi, 'b-', linewidth=2, label=f"NN Pred $\psi(x)$ (E={pred_energy:.3f})")
plt.fill_between(x_grid, exact_psi, pred_psi, color='red', alpha=0.2, label="Difference")

# 퍼텐셜 형태를 가이드로 옅게 표시 (스케일 조정)
V_scaled = (V_harmonic / np.max(V_harmonic)) * np.max(exact_psi)
plt.plot(x_grid, V_scaled, 'gray', linestyle=':', alpha=0.5, label="V(x) shape (scaled)")

plt.title("Comparison: Theoretical vs NN Harmonic Oscillator Ground State", fontsize=13)
plt.xlabel("Position x", fontsize=11)
plt.ylabel("$\psi(x)$", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right", fontsize=10)
plt.show()