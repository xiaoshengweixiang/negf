import numpy as np
import matplotlib.pyplot as plt


def conductance(H00, H01, N=5, W=[0], Ef=[0, 3], num=300, time=25):
    Ef = np.linspace(Ef[0], Ef[1], num)
    H10 = H01.conj().T
    res = []
    for w in W:
        T_LR = []
        Him = np.diag(np.random.rand(H00.shape[0]) - 0.5) * w
        H00_copy = H00 + Him
        for i in range(num):
            E = Ef[i] + 1j * 1e-6
            ai = H01.copy()
            bi = H01.conj().T.copy()
            ei = H00_copy.copy()
            eg = H00_copy.copy()
            for _ in range(time):
                mm = np.linalg.inv(E * np.eye(H00.shape[0]) - ei)
                eg = eg + ai @ mm @ bi
                ei = ei + ai @ mm @ bi + bi @ mm @ ai
                ai = ai @ mm @ ai
                bi = bi @ mm @ bi

            gr = np.linalg.inv(E * np.eye(H00.shape[0]) - eg)
            hgh_R = H01 @ gr @ H01.conj().T

            ai = H10.copy()
            bi = H10.conj().T.copy()
            ei = H00_copy.copy()
            eg = H00_copy.copy()
            for _ in range(time):
                mm = np.linalg.inv(E * np.eye(H00.shape[0]) - ei)
                eg = eg + ai @ mm @ bi
                ei = ei + ai @ mm @ bi + bi @ mm @ ai
                ai = ai @ mm @ ai
                bi = bi @ mm @ bi

            gr = np.linalg.inv(E * np.eye(H00.shape[0]) - eg)
            hgh_L = H10 @ gr @ H10.conj().T

            GR_ii = np.linalg.inv(E * np.eye(H00.shape[0]) - H00_copy - hgh_L)
            GR_1j = GR_ii.copy()

            for _ in range(2, N + 1):
                GR_ii = np.linalg.inv(E * np.eye(H00.shape[0]) - H00_copy - H10 @ GR_ii @ H01)
                GR_1j = GR_1j @ H01 @ GR_ii

            GR_ii = np.linalg.inv(np.linalg.inv(GR_ii) - hgh_R)
            GR_1j = GR_1j + GR_1j @ hgh_R @ GR_ii

            T_R = 1j * (hgh_R - hgh_R.conj().T)
            T_L = 1j * (hgh_L - hgh_L.conj().T)

            T_LR.append(np.real(np.trace(T_L @ GR_1j @ T_R @ GR_1j.conj().T)))
            res.append(T_LR)
        plt.plot(Ef, T_LR)

    plt.xlabel('Ef/eV')
    plt.ylabel('T_{LR}/e^2/h')
    plt.legend([str(w) for w in W])
    plt.title('conductance')
    plt.show()
    return res


def bands(H00, H01, num=1001):
    Energy = np.zeros((H00.shape[0], num))
    k = np.linspace(-1, 1, num)
    for i in range(num):
        H = H00 + H01 * np.exp(1j * k[i] * np.pi) + H01.conj().T * np.exp(-1j * k[i] * np.pi)
        eigenvalue = np.linalg.eigvals(H)
        eigenvalue_sorted = np.sort(eigenvalue)[::-1]
        Energy[:, i] = np.real(eigenvalue_sorted)
    plt.figure()
    for i in range(H00.shape[0]):
        plt.plot(k, Energy[i, :], 'black')
    plt.xlabel('k/pi')
    plt.ylabel('E')
    plt.title('band')
    plt.show()
    return Energy
