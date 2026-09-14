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