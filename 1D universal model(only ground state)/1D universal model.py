import numpy as np
import matplotlib.pyplot as plt

# ----------------- 물리 상수 -----------------
hbar = 1.054e-34
m_e = 9.109e-31

# ----------------- 5가지 포텐셜 정의 -----------------
def Harmonic_Oscillator_potential(x, m, omega):
    return 0.5 * m * (omega**2) * (x**2)

def Infinite_Square_Well_Potential(x, L, V_max=1e-16):
    return np.where(np.abs(x) <= L / 2.0, 0.0, V_max)

def Finite_Square_Well_Potential(x, L, V0):
    return np.where(np.abs(x) <= L / 2.0, 0.0, V0)

def Quartic_Anharmonic_Oscillator_Potential(x, m, omega, alpha):
    return 0.5 * m * (omega**2) * (x**2) + alpha * (x**4)

def Gaussian_Potential(x, V0, alpha):
    return -V0 * np.exp(-alpha * (x**2))


# ==============================================================================
# 1. 데이터셋 생성 영역
# ==============================================================================
x_min = -1e-9
x_max = 1e-9
num_grid = 200
x = np.linspace(x_min, x_max, num_grid)
domain_width = x_max - x_min

total_samples = 500
V_list = []

current_count = 0
while current_count < total_samples:
    # 1. Harmonic
    omega_rand = np.random.uniform(0.5e15, 2.0e15)
    V_list.append(Harmonic_Oscillator_potential(x, m=m_e, omega=omega_rand))
    current_count += 1
    if current_count >= total_samples: break

    # 2. Infinite Well (V_max = 1e-16 J)
    L_inf = np.random.uniform(0.3 * domain_width, 0.8 * domain_width)
    V_list.append(Infinite_Square_Well_Potential(x, L=L_inf, V_max=1e-16))
    current_count += 1
    if current_count >= total_samples: break

    # 3. Finite Well
    L_fin = np.random.uniform(0.2 * domain_width, 0.7 * domain_width)
    V0_fin = np.random.uniform(0.1e-18, 1.5e-18)
    V_list.append(Finite_Square_Well_Potential(x, L=L_fin, V0=V0_fin))
    current_count += 1
    if current_count >= total_samples: break

    # 4. Quartic Anharmonic
    omega_anh = np.random.uniform(0.5e15, 1.5e15)
    alpha_anh = np.random.uniform(1e17, 5e18)
    V_list.append(Quartic_Anharmonic_Oscillator_Potential(x, m=m_e, omega=omega_anh, alpha=alpha_anh))
    current_count += 1
    if current_count >= total_samples: break

    # 5. Gaussian
    V0_g = np.random.uniform(0.5e-18, 3.0e-18)
    sigma_g = np.random.uniform(0.1 * domain_width, 0.3 * domain_width)
    V_list.append(Gaussian_Potential(x, V0=V0_g, alpha=1.0 / (2.0 * sigma_g**2)))
    current_count += 1
    if current_count >= total_samples: break

V = np.array(V_list)
print(f"[데이터셋 생성 완료] 총 샘플 수: {len(V)}")


# ==============================================================================
# 2. 기저 함수(Basis)를 통한 파동함수 합성 및 에너지 계산
# ==============================================================================
num_basis = 20  # cos 10개 + sin 10개

def guess_psi_from_A(A, x_grid):
    L_box = x_grid[-1] - x_grid[0]
    psi = np.zeros_like(x_grid)
    half_k = len(A) // 2  # 10
    
    # 1. Cosine 기저 (우함수)
    for k in range(half_k):
        k_val = (2 * k + 1) * np.pi / L_box
        psi += A[k] * np.cos(k_val * x_grid)
        
    # 2. Sine 기저 (기함수 및 오프셋 보정)
    for k in range(half_k):
        k_val = 2 * (k + 1) * np.pi / L_box
        psi += A[half_k + k] * np.sin(k_val * x_grid)
        
    return psi

def compute_energy_from_A(A, V_grid, x_grid, m=m_e):
    psi = guess_psi_from_A(A, x_grid)
    dx = x_grid[1] - x_grid[0]
    
    ddpsi = np.gradient(np.gradient(psi, dx), dx)
    T_psi = -(hbar**2 / (2.0 * m)) * ddpsi
    V_psi = V_grid * psi
    H_psi = T_psi + V_psi
    
    num = np.trapezoid(psi * H_psi, x_grid)
    den = np.trapezoid(psi * psi, x_grid)
    
    if den < 1e-18:
        return 1e10
    return num / den

def get_loss(A_cand, V_raw, x_grid, m=m_e):
    V_shifted = V_raw - np.min(V_raw)
    E = compute_energy_from_A(A_cand, V_shifted, x_grid, m=m)
    E_scale = 1.054e-34 * 1.0e15  # hbar * omega0
    return E / E_scale

# 기준 에너지 스케일 (약 1.0 eV = 1.602e-19 J 단위로 정규화하여 벽 높이 보존)
V_SCALE_REF = 1.602e-19 

def preprocess_V(V_raw_in):
    # 바닥을 0으로 맞추고 물리적 스케일(eV 단위)로 입력 정규화
    V_shifted = V_raw_in - np.min(V_raw_in)
    V_norm = V_shifted / V_SCALE_REF
    return np.clip(V_norm, 0.0, 20.0)  # 무한 장벽 극단치 컷오프


# ==============================================================================
# 3. 범용 신경망
# ==============================================================================
class NNVM_Basis:
    def __init__(self, in_dim=200, out_dim=20, hidden_layers=[128, 64, 64], seed=42):
        if seed is not None:
            np.random.seed(seed)
        dims = [in_dim] + hidden_layers + [out_dim]
        self.num_layers = len(dims) - 1
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers):
            limit = np.sqrt(6.0 / (dims[i] + dims[i+1]))
            W = np.random.uniform(-limit, limit, (dims[i], dims[i+1]))
            b = np.zeros((1, dims[i+1]))
            self.weights.append(W)
            self.biases.append(b)
            
        self.activations = []

    def _act(self, z):
        return np.tanh(z)

    def _act_deriv(self, a):
        return 1.0 - a**2

    def forward(self, V_input):
        a = V_input.reshape(1, -1) if V_input.ndim == 1 else V_input
        self.activations = [a]
        
        for i in range(self.num_layers - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self._act(z)
            self.activations.append(a)
            
        z_last = np.dot(a, self.weights[-1]) + self.biases[-1]
        self.activations.append(z_last)
        return z_last.flatten()

    def backward(self, loss_func, eps=1e-5):
        A_out = self.activations[-1].flatten().copy()
        dL_dA = np.zeros_like(A_out)
        
        for i in range(len(A_out)):
            A_p = A_out.copy(); A_p[i] += eps
            A_m = A_out.copy(); A_m[i] -= eps
            dL_dA[i] = (loss_func(A_p) - loss_func(A_m)) / (2.0 * eps)
            
        delta = dL_dA.reshape(1, -1)
        grads_W = [None] * self.num_layers
        grads_b = [None] * self.num_layers
        
        for l in range(self.num_layers - 1, -1, -1):
            a_prev = self.activations[l]
            grads_W[l] = np.dot(a_prev.T, delta)
            grads_b[l] = delta
            if l > 0:
                delta = np.dot(delta, self.weights[l].T) * self._act_deriv(a_prev)
                
        return grads_W, grads_b


# ==============================================================================
# 4. 학습 루프
# ==============================================================================
model = NNVM_Basis(in_dim=num_grid, out_dim=num_basis, hidden_layers=[128, 64, 64], seed=42)

epochs = 100
learning_rate = 0.005

print("\n" + "=" * 60)
print(f" 범용 기저 계수 추측 신경망 학습 시작 (에폭: {epochs})")
print("=" * 60)

for epoch in range(epochs):
    epoch_loss = 0.0
    indices = np.random.permutation(len(V))
    
    for idx in indices:
        V_raw_sample = V[idx]
        
        # 1. 전처리 (실제 벽 높이 정보를 반영한 입력)
        V_norm_input = preprocess_V(V_raw_sample)
        
        # 2. 순전파
        A_pred = model.forward(V_norm_input)
        
        # 3. 손실 함수 및 역전파
        loss_fn = lambda A: get_loss(A, V_raw_sample, x, m=m_e)
        epoch_loss += loss_fn(A_pred)
        
        grads_W, grads_b = model.backward(loss_fn, eps=1e-5)
        for l in range(model.num_layers):
            model.weights[l] -= learning_rate * grads_W[l]
            model.biases[l] -= learning_rate * grads_b[l]
            
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] - Average Loss: {epoch_loss / len(V):.6f}")

print("\n[학습 완료]\n")


# ==============================================================================
# 5. 유한퍼텐셜 우물 검증 (FD Matrix 수치해와 비교)
# ==============================================================================
L_test = 0.8e-9
V0_test = 2.0e-19  # 약 1.25 eV
V_test_raw = Finite_Square_Well_Potential(x, L=L_test, V0=V0_test)

# 모델 추론
V_test_norm = preprocess_V(V_test_raw)
A_pred_raw = model.forward(V_test_norm)
psi_guessed = guess_psi_from_A(A_pred_raw, x)

# 규격화
norm_val = np.sqrt(np.trapezoid(psi_guessed**2, x))
if norm_val > 1e-18:
    psi_guessed = psi_guessed / norm_val
if psi_guessed[len(psi_guessed) // 2] < 0:
    psi_guessed = -psi_guessed

# FD Matrix 고윳값 대각화 (100% 참값)
dx = x[1] - x[0]
T_diag = np.ones(len(x)) * (hbar**2 / (m_e * (dx**2)))
T_off = np.ones(len(x) - 1) * (-hbar**2 / (2.0 * m_e * (dx**2)))
H_mat = np.diag(T_diag + V_test_raw) + np.diag(T_off, 1) + np.diag(T_off, -1)

evals, evecs = np.linalg.eigh(H_mat)
E_exact = evals[0]
psi_exact = evecs[:, 0]
if psi_exact[len(psi_exact) // 2] < 0:
    psi_exact = -psi_exact
psi_exact = psi_exact / np.sqrt(np.trapezoid(psi_exact**2, x))

E_pred = compute_energy_from_A(A_pred_raw, V_test_raw, x, m=m_e)
rel_error = abs(E_pred - E_exact) / E_exact * 100.0

# 결과 출력
print("=" * 60)
print(" [유한퍼텐셜 우물 바닥상태 예측 검증]")
print(f" 우물 폭 (L)                      : {L_test * 1e9:.2f} nm")
print(f" 장벽 높이 (V0)                   : {V0_test / 1.602e-19:.3f} eV")
print(f" 엄밀한 바닥상태 에너지 (Exact E0) : {E_exact:.6e} J ({E_exact / 1.602e-19:.3f} eV)")
print(f" 모델 예측 바닥상태 에너지 (Pred E0) : {E_pred:.6e} J ({E_pred / 1.602e-19:.3f} eV)")
print(f" 에너지 상대 오차율               : {rel_error:.4f} %")
print("=" * 60)

# 시각화
plt.figure(figsize=(9, 5))
x_nm = x * 1e9

plt.plot(x_nm, psi_exact, label=r'Exact Ground State $\psi_0(x)$', color='black', linestyle='--', linewidth=2.0)
plt.plot(x_nm, psi_guessed, label=r'Model Pred $\psi(x)$', color='darkorange', linewidth=1.8)
plt.fill_between(x_nm, psi_exact, psi_guessed, color='orange', alpha=0.25, label='Difference')

plt.axvline(-L_test / 2.0 * 1e9, color='gray', linestyle=':', label='Well Boundary')
plt.axvline(L_test / 2.0 * 1e9, color='gray', linestyle=':')

plt.title(f"Finite Square Well Ground State ($L = {L_test * 1e9:.2f}\\ \\mathrm{{nm}}$, $V_0 = {V0_test / 1.602e-19:.2f}\\ \\mathrm{{eV}}$)", fontsize=13, pad=10)
plt.xlabel("Position $x$ (nm)", fontsize=11)
plt.ylabel(r"$\psi(x)$", fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10, loc='upper right')
plt.xlim(x_nm[0], x_nm[-1])
plt.tight_layout()

plt.show()