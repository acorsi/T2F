
import h5py, numpy as np
f = h5py.File("../drift_output/drift_G4_40mu_merged.h5", "r")
pid = np.array(f["H"]["pid"][0]).astype(int)
n = len(pid)
print("liste répétée (2 moitiés identiques) :", n % 2 == 0 and np.array_equal(pid[:n//2], pid[n//2:]))
print("doublons adjacents :", np.sum(pid[1:] == pid[:-1]))
