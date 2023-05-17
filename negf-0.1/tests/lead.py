from negf import itgf
import numpy as np

M = 5
t = 0.1
H00 = np.eye(M) * 4 * t - np.diag(np.ones(M - 1), 1) * t - np.diag(np.ones(M - 1), -1) * t
H01 = - t * np.eye(M)
W = [0, 0.2, 0.4, 0.8]
Ef = [0, 2]
trans = itgf.conductance(H00, H01, M, W, Ef)
bands = itgf.bands(H00, H01)
