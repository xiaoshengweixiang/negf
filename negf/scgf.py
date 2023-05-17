import numpy as np
import matplotlib.pyplot as plt

# Constants
N = 3
M = 5
state = 4
num = 300
t1 = 0.7
t2 = 0.4
tin = 0.3
tsoc = 0.9
lambdamag = -0.4

# Pauli matrices
sigma0 = np.eye(2)
sigmaX = np.array([[0, 1], [1, 0]])
sigmaY = np.array([[0, -1j], [1j, 0]])
sigmaZ = np.array([[1, 0], [0, -1]])

tauX = np.array([[0, 1], [1, 0]])
tauY = np.array([[0, -1j], [1j, 0]])
tauZ = np.array([[1, 0], [0, -1]])

# Matrices A through G
A = 0.5*t1*np.kron(tauX, sigma0) - 0.5j*t1*np.kron(tauY, sigma0) - tin*np.kron(tauZ, sigmaZ) - 0.5j*tsoc*np.kron(tauZ, sigmaY)
B = 0.5*t1*np.kron(tauX, sigma0) + 0.5j*t1*np.kron(tauY, sigma0) - tin*np.kron(tauZ, sigmaZ) + 0.5j*tsoc*np.kron(tauZ, sigmaY)
C = 0.5*t2*np.kron(tauX, sigma0) + 0.5j*t2*np.kron(tauY, sigma0) - tin*np.kron(tauZ, sigmaZ) + 0.5j*tsoc*np.kron(tauZ, sigmaX)
K = 0.5*t2*np.kron(tauX, sigma0) - 0.5j*t2*np.kron(tauY, sigma0) - tin*np.kron(tauZ, sigmaZ) - 0.5j*tsoc*np.kron(tauZ, sigmaX)
E = 0.5*t2*np.kron(tauX, sigma0) + 0.5j*t2*np.kron(tauY, sigma0)
F = 0.5*t2*np.kron(tauX, sigma0) - 0.5j*t2*np.kron(tauY, sigma0)
G = t1*np.kron(tauX, sigma0) + lambdamag*np.kron(tauZ, sigmaZ)

# Effective Hamiltonian
H00 = np.kron(np.eye(N), G) + np.kron(np.diag(np.ones(N-1), 1), C) + np.kron(np.diag(np.ones(N-1), -1), K)
H01 = np.kron(np.eye(N), A) + np.kron(np.diag(np.ones(N-1), -1), F)

# Disorder list
L = [0]
Ef = np.linspace(0, 3, num)

plt.figure()
for w in L:
    Him = np.diag(np.random.rand(4*N)-0.5)*w
    H00 += Him
    T_LR = []
    for i in range(num):
        E = Ef[i] + 1j * 1e-6
        time = 25
        ai = H01.copy()
        bi = H01.conj().T.copy()
        ei = H00.copy()
        eg = H00.copy()
        for _ in range(time):
            mm = np.linalg.inv(E * np.eye(state * N) - ei)
            eg += ai @ mm @ bi
            ei += ai @ mm @ bi + bi @ mm @ ai
            ai = ai @ mm @ ai
            bi = bi @ mm @ bi
        gr = np.linalg.inv(E * np.eye(state * N) - eg)
        hgh_R = H01 @ gr @ H01.conj().T

        H10 = H01.conj().T
        ai = H10.copy()
        bi = H10.conj().T.copy()
        ei = H00.copy()
        eg = H00.copy()
        for _ in range(time):
            mm = np.linalg.inv(E * np.eye(state * N) - ei)
            eg += ai @ mm @ bi
            ei += ai @ mm @ bi + bi @ mm @ ai
            ai = ai @ mm @ ai
            bi = bi @ mm @ bi
        gr = np.linalg.inv(E * np.eye(state * N) - eg)
        hgh_L = H10 @ gr @ H10.conj().T

        GR_ii = np.linalg.inv(E * np.eye(state * N) - H00 - hgh_L)
        GR_1j = GR_ii.copy()
        for _ in range(2, M + 1):
            GR_ii = np.linalg.inv(E * np.eye(state * N) - H00 - H10 @ GR_ii @ H01)
            GR_1j = GR_1j @ H01 @ GR_ii
        GR_ii = np.linalg.inv(np.linalg.inv(GR_ii) - hgh_R)
        GR_1j = GR_1j + GR_1j @ hgh_R @ GR_ii

        T_R = 1j * (hgh_R - hgh_R.conj().T)
        T_L = 1j * (hgh_L - hgh_L.conj().T)
        T_LR.append(np.real(np.trace(T_L @ GR_1j @ T_R @ GR_1j.conj().T)))
    plt.plot(Ef, T_LR, label=str(w))
plt.xlabel('Ef/eV')
plt.ylabel('T_{LR}/e^2/h')
plt.legend(title='Disorder')
plt.title('Conductance')
plt.show()