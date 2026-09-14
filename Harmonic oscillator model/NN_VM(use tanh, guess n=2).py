import numpy as np
import matplotlib.pyplot as plt

# -----------------Harmonic Oscillator의 에너지 고유값을 계산하는 함수------------------------------

def Harmonic_Oscilator_Energyeigenvalues(psi_func):
    m = 9.09 * 10**(-31)
    hbar = 1.054 * 10**(-34)
    omega = 1.0 * 10**(15)
    x_lim = 1.0 * 10**(-9)
    split_num = 1000
    
    x = np.linspace(-x_lim, x_lim, split_num)
    dx = x[1] - x[0]
    
    psi = psi_func(x)
    ddpsi = np.gradient(np.gradient(psi, dx), dx)
    
    T_psi = -(hbar**2 / (2 * m)) * ddpsi
    V_psi = 0.5 * m * (omega**2) * (x**2) * psi
    H_psi = T_psi + V_psi
    
    numerator = np.trapezoid(psi * H_psi, x)
    denominator = np.trapezoid(psi * psi, x)
    
    E = numerator / denominator
    return E

# -----------------이미 알려진 Harmonic Oscillator의 Ground State Wavefunction------------------------------

def psi_ground (x) :
    m = 9.09 * 10**(-31)
    hbar = 1.054 * 10**(-34)
    omega = 1.0 * 10**(15)

    return ((m * omega) / (hbar * np.pi))**0.25 * np.exp(-0.5 * (m * omega / hbar) * (x**2))


##-----------------NN의 output을 입력받아 파동함수로 만들어주는 함수------------------------------

def guess_psi(A):

    A = np.array(A, dtype=float)
    
    def psi(x):

        decay_width = 3.405e-10
        L = 1.0e-9
        envelope = np.exp(-0.5 * (x / decay_width)**2)
        
        fourier_sum = np.zeros_like(x, dtype=float)
        for n_idx, coeff in enumerate(A):
    
            n = n_idx
            fourier_sum += coeff * np.cos((2 * n + 1) * np.pi * x / (2 * L))
            
        return envelope * fourier_sum

    return psi

##-----------------NN------------------------------

class NNVM:
    def __init__(self, layer_sizes=[1, 10, 10, 10, 10, 10], seed=42):
        if seed is not None:
            np.random.seed(seed)
            
        self.num_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            W = np.random.uniform(-limit, limit, (in_dim, out_dim))
            b = np.zeros((1, out_dim))
            self.weights.append(W)
            self.biases.append(b)
            
        self.activations = []
        self.z_values = []

    def _act(self, z):
        # tanh 활성화 함수
        return np.tanh(z)

    def _act_derivative(self, a):
        # tanh 도함수: d(tanh(z))/dz = 1 - a^2
        return 1.0 - a**2

    def forward(self, x=None):
        if x is None:
            x = np.array([[1.0]])
            
        self.activations = [x]
        self.z_values = []
        
        a = x
        # 1. 은닉층: tanh 적용
        for i in range(self.num_layers - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self._act(z)
            self.z_values.append(z)
            self.activations.append(a)
            
        # 2. 마지막 출력층: Linear 적용 (tanh 포화 현상 방지)
        z_last = np.dot(a, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_last)
        self.activations.append(z_last)
            
        return z_last.flatten()

    def loss(self, predicted):
        # 파동함수 생성 후 조화진동자 에너지 고윳값 계산
        E = Harmonic_Oscilator_Energyeigenvalues(guess_psi(predicted))
        E_scale = 1.054e-34 * 1.0e15  # hbar * omega 로 정규화
        return E / E_scale

    def backward(self, loss_func, eps=1e-4):
        A = self.activations[-1].flatten().copy()
        num_outputs = len(A)
        dL_dout = np.zeros(num_outputs)
        
        for i in range(num_outputs):
            A_plus = A.copy()
            A_minus = A.copy()
            
            A_plus[i] += eps
            A_minus[i] -= eps
            
            E_plus = loss_func(A_plus)
            E_minus = loss_func(A_minus)
            
            dL_dout[i] = (E_plus - E_minus) / (2.0 * eps)
            
        dL_dout = dL_dout.reshape(1, -1)
        grads_W = [None] * self.num_layers
        grads_b = [None] * self.num_layers
        
        # 마지막 층은 Linear이므로 delta = dL_dout
        delta = dL_dout
        
        # 은닉층 역전파 진행
        for l in range(self.num_layers - 1, -1, -1):
            a_prev = self.activations[l]
            
            grads_W[l] = np.dot(a_prev.T, delta)
            grads_b[l] = delta
            
            if l > 0:
                delta = np.dot(delta, self.weights[l].T) * self._act_derivative(a_prev)
                
        return grads_W, grads_b

#-----------------즐거운 학습의 시간-----------------------------
model = NNVM(seed=42)

lr = 0.01
epochs = 1000


for epoch in range(epochs):
    
    A = model.forward()
    
    current_loss = model.loss(A)
    
    grads_W, grads_b = model.backward(model.loss)
    
    for l in range(model.num_layers):
        model.weights[l] -= lr * grads_W[l]
        model.biases[l] -= lr * grads_b[l]
        
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Energy (Loss): {current_loss:.6e} J")

final_A = model.forward()

guessed_psi = guess_psi(final_A)

ground_state_energy = Harmonic_Oscilator_Energyeigenvalues(psi_ground)
guessed_energy = Harmonic_Oscilator_Energyeigenvalues(guessed_psi)

print(ground_state_energy, guessed_energy)


# ==========================================
# 1. 시각화 데이터 계산 및 정규화
# ==========================================
#x_plot = np.linspace(-1e-9, 1e-9, 1000)
#dx_plot = x_plot[1] - x_plot[0]

# 이론적 파동함수
#psi_exact = psi_ground(x_plot)

# 신경망 파동함수
#psi_pred = guessed_psi(x_plot)

# 신경망 출력 파동함수 수치 정규화: <psi|psi> = 1
#norm_pred = np.sqrt(np.trapezoid(psi_pred**2, x_plot))
#if norm_pred > 1e-15:
#    psi_pred = psi_pred / norm_pred

# 전체 위상(부호) 정렬: 원점(x=0)에서 양수가 되도록 설정
#if psi_pred[len(psi_pred) // 2] < 0:
#    psi_pred = -psi_pred

# ==========================================
# 2. Matplotlib 플롯 생성
# ==========================================
#plt.figure(figsize=(10, 6))

# 나노미터(nm) 단위로 x축 변환
#x_nm = x_plot * 1e9

# 파동함수 비교 곡선
#plt.plot(x_nm, psi_exact, label='Theoretical $\psi_0(x)$', color='black', linewidth=2.0, linestyle='--')
#plt.plot(x_nm, psi_pred, label='NN Guessed $\psi(x)$', color='royalblue', linewidth=1.8, alpha=0.85)

# 오차 영역 음영 표시
#plt.fill_between(x_nm, psi_exact, psi_pred, color='salmon', alpha=0.3, label='Difference')

# 그래프 스타일링
#plt.title("Comparison of Ground State Wavefunctions", fontsize=14, pad=12)
#plt.xlabel("Position $x$ (nm)", fontsize=12)
#plt.ylabel("$\psi(x)$ ($\mathrm{m}^{-1/2}$)", fontsize=12)
#plt.grid(True, linestyle=':', alpha=0.6)
#plt.legend(fontsize=11, loc='upper right')
#plt.xlim(-1.0, 1.0)
#plt.tight_layout()

#plt.show()

# ==============================================================================
# 1차 여기상태(n=1, First Excited State) 파동함수 추측 알고리즘
# ==============================================================================

# -----------------1차 여기상태를 위한 파동함수 생성 함수 (홀수 대칭성 고려)-----------------
def guess_psi_first_excited(A):
    A = np.array(A, dtype=float)
    
    def psi(x):
        decay_width = 3.405e-10
        L = 1.0e-9
        envelope = np.exp(-0.5 * (x / decay_width)**2)
        
        # n=1 상태는 기함수(Odd parity) 형태이므로 sin 기저를 사용
        fourier_sum = np.zeros_like(x, dtype=float)
        for n_idx, coeff in enumerate(A):
            n = n_idx + 1
            fourier_sum += coeff * np.sin(n * np.pi * x / L)
            
        return envelope * fourier_sum

    return psi

# -----------------이론적인 1차 여기상태 파동함수-----------------
def psi_first_excited_exact(x):
    m = 9.09 * 10**(-31)
    hbar = 1.054 * 10**(-34)
    omega = 1.0 * 10**(15)
    alpha = m * omega / hbar
    
    # H_1(y) = 2y (Hermite Polynomial)
    y = np.sqrt(alpha) * x
    psi_0 = (alpha / np.pi)**0.25 * np.exp(-0.5 * alpha * (x**2))
    return (1.0 / np.sqrt(2.0)) * (2.0 * y) * psi_0

# -----------------1차 여기상태 전용 손실 함수 정의-----------------
# 손실함수 = 정규화된 에너지 고윳값 + lambda_ortho * (바닥상태와의 겹침 적분)^2
def loss_first_excited(A, target_psi_ground=psi_ground, lambda_ortho=10.0):
    psi_cand = guess_psi_first_excited(A)
    
    # 1. 에너지 기댓값 계산
    E = Harmonic_Oscilator_Energyeigenvalues(psi_cand)
    E_scale = 1.054e-34 * 1.0e15  # hbar * omega 로 정규화
    normalized_E = E / E_scale
    
    # 2. 바닥상태와의 정규화된 겹침(Overlap / Inner product) 계산
    x_grid = np.linspace(-1.0e-9, 1.0e-9, 1000)
    psi_0_vals = target_psi_ground(x_grid)
    psi_1_vals = psi_cand(x_grid)
    
    norm_0 = np.sqrt(np.trapezoid(psi_0_vals**2, x_grid))
    norm_1 = np.sqrt(np.trapezoid(psi_1_vals**2, x_grid))
    
    if norm_0 > 1e-15 and norm_1 > 1e-15:
        overlap = np.trapezoid(psi_0_vals * psi_1_vals, x_grid) / (norm_0 * norm_1)
    else:
        overlap = 0.0
        
    # 3. 총 손실 (에너지 최소화 + 직교성 페널티)
    total_loss = normalized_E + lambda_ortho * (overlap**2)
    return total_loss

# -----------------1차 여기상태 신경망 모델 초기화 및 학습-----------------
model_excited = NNVM(layer_sizes=[1, 10, 10, 10, 10, 10], seed=100)

lr_excited = 0.01
epochs_excited = 1000

print("\n--- 1차 여기상태(n=1) 학습 시작 ---")
for epoch in range(epochs_excited):
    A_curr = model_excited.forward()
    
    current_loss = loss_first_excited(A_curr, target_psi_ground=psi_ground)
    
    grads_W, grads_b = model_excited.backward(lambda a: loss_first_excited(a, target_psi_ground=psi_ground))
    
    for l in range(model_excited.num_layers):
        model_excited.weights[l] -= lr_excited * grads_W[l]
        model_excited.biases[l] -= lr_excited * grads_b[l]
        
    if (epoch + 1) % 200 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs_excited}] - Total Loss: {current_loss:.6f}")

# -----------------결과 검증 및 에너지 출력-----------------
final_A_excited = model_excited.forward()
guessed_psi_1 = guess_psi_first_excited(final_A_excited)

exact_E1 = Harmonic_Oscilator_Energyeigenvalues(psi_first_excited_exact)
guessed_E1 = Harmonic_Oscilator_Energyeigenvalues(guessed_psi_1)
hbar_omega = 1.054e-34 * 1.0e15

print("\n[에너지 고윳값 비교]")
print(f"이론적 1차 여기상태 에너지 (1.5 hbar*omega): {exact_E1:.6e} J ({exact_E1 / hbar_omega:.4f} ħω)")
print(f"추측된 1차 여기상태 에너지                 : {guessed_E1:.6e} J ({guessed_E1 / hbar_omega:.4f} ħω)")

# ==============================================================================
# 2차 여기상태(n=2, Second Excited State) 파동함수 추측 알고리즘
# ==============================================================================

# -----------------2차 여기상태를 위한 파동함수 생성 함수 (짝수 대칭성 고려)-----------------
def guess_psi_second_excited(A):
    A = np.array(A, dtype=float)
    
    def psi(x):
        decay_width = 3.405e-10
        L = 1.0e-9
        envelope = np.exp(-0.5 * (x / decay_width)**2)
        
        # n=2 상태는 우함수(Even parity) 형태이므로 cos 기저를 사용
        fourier_sum = np.zeros_like(x, dtype=float)
        for n_idx, coeff in enumerate(A):
            n = n_idx
            fourier_sum += coeff * np.cos((2 * n + 1) * np.pi * x / (2 * L))
            
        return envelope * fourier_sum

    return psi

# -----------------이론적인 2차 여기상태 파동함수-----------------
def psi_second_excited_exact(x):
    m = 9.09 * 10**(-31)
    hbar = 1.054 * 10**(-34)
    omega = 1.0 * 10**(15)
    alpha = m * omega / hbar
    
    # H_2(y) = 4y^2 - 2 (Hermite Polynomial)
    y = np.sqrt(alpha) * x
    psi_0 = (alpha / np.pi)**0.25 * np.exp(-0.5 * alpha * (x**2))
    return (1.0 / np.sqrt(8.0)) * (4.0 * (y**2) - 2.0) * psi_0

# -----------------2차 여기상태 전용 손실 함수 정의-----------------
# 손실함수 = 정규화된 에너지 고윳값 + lambda_ortho * (<psi_0|psi_2>^2 + <psi_1|psi_2>^2)
def loss_second_excited(A, target_psi_0=psi_ground, target_psi_1=guessed_psi_1, lambda_ortho=10.0):
    psi_cand = guess_psi_second_excited(A)
    
    # 1. 에너지 기댓값 계산
    E = Harmonic_Oscilator_Energyeigenvalues(psi_cand)
    E_scale = 1.054e-34 * 1.0e15  # hbar * omega 로 정규화
    normalized_E = E / E_scale
    
    # 2. 바닥상태(n=0) 및 1차 여기상태(n=1)와의 정규화된 겹침 적분 계산
    x_grid = np.linspace(-1.0e-9, 1.0e-9, 1000)
    psi_0_vals = target_psi_0(x_grid)
    psi_1_vals = target_psi_1(x_grid)
    psi_2_vals = psi_cand(x_grid)
    
    norm_0 = np.sqrt(np.trapezoid(psi_0_vals**2, x_grid))
    norm_1 = np.sqrt(np.trapezoid(psi_1_vals**2, x_grid))
    norm_2 = np.sqrt(np.trapezoid(psi_2_vals**2, x_grid))
    
    overlap_0 = 0.0
    overlap_1 = 0.0
    
    if norm_2 > 1e-15:
        if norm_0 > 1e-15:
            overlap_0 = np.trapezoid(psi_0_vals * psi_2_vals, x_grid) / (norm_0 * norm_2)
        if norm_1 > 1e-15:
            overlap_1 = np.trapezoid(psi_1_vals * psi_2_vals, x_grid) / (norm_1 * norm_2)
            
    # 3. 총 손실 (에너지 최소화 + 직교성 페널티)
    total_loss = normalized_E + lambda_ortho * (overlap_0**2 + overlap_1**2)
    return total_loss

# -----------------2차 여기상태 신경망 모델 초기화 및 학습-----------------
model_second_excited = NNVM(layer_sizes=[1, 10, 10, 10, 10, 10], seed=200)

lr_second = 0.01
epochs_second = 1000

print("\n--- 2차 여기상태(n=2) 학습 시작 ---")
for epoch in range(epochs_second):
    A_curr = model_second_excited.forward()
    
    current_loss = loss_second_excited(A_curr, target_psi_0=psi_ground, target_psi_1=guessed_psi_1)
    
    grads_W, grads_b = model_second_excited.backward(
        lambda a: loss_second_excited(a, target_psi_0=psi_ground, target_psi_1=guessed_psi_1)
    )
    
    for l in range(model_second_excited.num_layers):
        model_second_excited.weights[l] -= lr_second * grads_W[l]
        model_second_excited.biases[l] -= lr_second * grads_b[l]
        
    if (epoch + 1) % 200 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs_second}] - Total Loss: {current_loss:.6f}")

# -----------------결과 검증 및 에너지 출력-----------------
final_A_second = model_second_excited.forward()
guessed_psi_2 = guess_psi_second_excited(final_A_second)

exact_E2 = Harmonic_Oscilator_Energyeigenvalues(psi_second_excited_exact)
guessed_E2 = Harmonic_Oscilator_Energyeigenvalues(guessed_psi_2)
hbar_omega = 1.054e-34 * 1.0e15

print("\n[에너지 고윳값 비교]")
print(f"이론적 2차 여기상태 에너지 (2.5 hbar*omega): {exact_E2:.6e} J ({exact_E2 / hbar_omega:.4f} ħω)")
print(f"추측된 2차 여기상태 에너지                 : {guessed_E2:.6e} J ({guessed_E2 / hbar_omega:.4f} ħω)")

# ==========================================
# n=2 파동함수 시각화 데이터 계산 및 정규화
# ==========================================
x_plot = np.linspace(-1e-9, 1e-9, 1000)
x_nm = x_plot * 1e9

# 정규화 함수
def get_normalized_curve(psi_fn, x):
    y = psi_fn(x)
    norm = np.sqrt(np.trapezoid(y**2, x))
    return y / norm if norm > 1e-15 else y

# 이론적 파동함수 및 추측 파동함수 정규화
psi2_exact_plot = get_normalized_curve(psi_second_excited_exact, x_plot)
psi2_pred_plot = get_normalized_curve(guessed_psi_2, x_plot)

# 위상(부호) 정렬: 원점(x=0)에서 이론값과 동일하게 음수가 되도록 설정 (H_2(0) < 0)
if psi2_pred_plot[len(psi2_pred_plot) // 2] > 0:
    psi2_pred_plot = -psi2_pred_plot

# ==========================================
# Matplotlib 플롯 생성
# ==========================================
plt.figure(figsize=(10, 6))

# 기준선으로 사용할 이전 상태들 (흐리게 표시)
psi0_plot = get_normalized_curve(psi_ground, x_plot)
psi1_plot = get_normalized_curve(guessed_psi_1, x_plot)
plt.plot(x_nm, psi0_plot, label='$n=0\ (\psi_0)$', color='gray', linestyle=':', alpha=0.5)
plt.plot(x_nm, psi1_plot, label='$n=1\ (\psi_1)$', color='gray', linestyle='--', alpha=0.5)

# n=2 이론값 및 추측값 비교
plt.plot(x_nm, psi2_exact_plot, label='Exact Excited $\psi_2(x)$', color='black', linewidth=2.0, linestyle='--')
plt.plot(x_nm, psi2_pred_plot, label='NN Guessed $\psi_2(x)$', color='darkorange', linewidth=1.8, alpha=0.85)

# 오차 영역 음영 표시
plt.fill_between(x_nm, psi2_exact_plot, psi2_pred_plot, color='orange', alpha=0.2, label='Difference')

# 그래프 스타일링
plt.title("Harmonic Oscillator $n=2$ Wavefunction Guess", fontsize=14, pad=12)
plt.xlabel("Position $x$ (nm)", fontsize=12)
plt.ylabel("$\psi(x)$ ($\mathrm{m}^{-1/2}$)", fontsize=12)
plt.axhline(0, color='black', linewidth=0.6, linestyle='-')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11, loc='upper right')
plt.xlim(-1.0, 1.0)
plt.tight_layout()

plt.show()