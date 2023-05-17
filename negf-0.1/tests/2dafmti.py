from negf import itgf
import numpy as np

# Constants
M = 5
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
A = 0.5 * t1 * np.kron(tauX, sigma0) - 0.5j * t1 * np.kron(tauY, sigma0) - tin * np.kron(tauZ,
                                                                                         sigmaZ) - 0.5j * tsoc * np.kron(
    tauZ, sigmaY)
B = 0.5 * t1 * np.kron(tauX, sigma0) + 0.5j * t1 * np.kron(tauY, sigma0) - tin * np.kron(tauZ,
                                                                                         sigmaZ) + 0.5j * tsoc * np.kron(
    tauZ, sigmaY)
C = 0.5 * t2 * np.kron(tauX, sigma0) + 0.5j * t2 * np.kron(tauY, sigma0) - tin * np.kron(tauZ,
                                                                                         sigmaZ) + 0.5j * tsoc * np.kron(
    tauZ, sigmaX)
K = 0.5 * t2 * np.kron(tauX, sigma0) - 0.5j * t2 * np.kron(tauY, sigma0) - tin * np.kron(tauZ,
                                                                                         sigmaZ) - 0.5j * tsoc * np.kron(
    tauZ, sigmaX)
E = 0.5 * t2 * np.kron(tauX, sigma0) + 0.5j * t2 * np.kron(tauY, sigma0)
F = 0.5 * t2 * np.kron(tauX, sigma0) - 0.5j * t2 * np.kron(tauY, sigma0)
G = t1 * np.kron(tauX, sigma0) + lambdamag * np.kron(tauZ, sigmaZ)

# Effective Hamiltonian
H00 = np.kron(np.eye(M), G) + np.kron(np.diag(np.ones(M - 1), 1), C) + np.kron(np.diag(np.ones(M - 1), -1), K)
H01 = np.kron(np.eye(M), A) + np.kron(np.diag(np.ones(M - 1), -1), F)
# Disorder
W = [0, 0.1, 0.3, 0.5]
trans = itgf.conductance(H00, H01, M, W)
bands = itgf.bands(H00, H01)
