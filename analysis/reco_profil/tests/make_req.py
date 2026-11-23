"""
make_req.py : fabrique une requete SYNTHETIQUE pour profile_basis / profile_basis_fast (aucune donnee reelle).
Les pads sont numerotes comme dans PadPlane::build_anode (zone quinconce, n_rings = 0) avec les parametres de drift.cfg.
Usage : python3 make_req.py requete.txt [n_evenements] [graine]
"""
import sys
import numpy as np

pad_size, inter_pad, r_quinc, r_active = 0.5, 0.015, 10.0, 10.0      # drift.cfg
pitch = pad_size + inter_pad
half = pad_size / 2.0

def pads():
    out = []
    n = int(np.ceil(r_active / pitch)) + 1
    for row in range(-n, n + 1):
        if row == 0:
            continue
        y = row * pitch - pitch / 2.0 if row > 0 else row * pitch + pitch / 2.0
        xo = pitch / 2.0 if row % 2 != 0 else 0.0     # row % 2 != 0 comme en C++ (valable pour row < 0)
        for col in range(-n, n + 1):
            x = col * pitch + xo
            if np.hypot(abs(x) + half, abs(y) + half) < r_quinc:
                out.append((x, y))
    return np.array(out)

def main():
    path = sys.argv[1]
    nev = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rng = np.random.RandomState(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    wide = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5        # distance max pad-trace, en pitchs
    P = pads()
    step_eff = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
    delta = float(sys.argv[6]) if len(sys.argv) > 6 else 0.2
    fastfmt = len(sys.argv) > 7 and sys.argv[7] == "fast"       # format etendu : masque e % 3 + 1 par evenement
    with open(path, "w") as f:
        if fastfmt:
            f.write("#fast 1\n")
        f.write("%.17g %.17g %d\n" % (step_eff, delta, nev))
        for e in range(nev):
            v = np.array([rng.uniform(-1, 1), 0.0, rng.uniform(1, 19.5)])      # z proche de la cathode ou de la grille aussi
            us, ls = [], []
            for k in range(2):
                sg = 1.0 if k == 0 else -1.0
                a = rng.uniform(0.0, 2 * np.pi)
                w = np.array([np.cos(a), sg * abs(np.sin(a)) + sg * 0.05, rng.uniform(-1.0, 1.0)])   # uz de signe quelconque
                w = w / np.linalg.norm(w)
                us.append(w)
                ls.append(rng.choice([0.05, 0.19, 0.2, 0.21, 0.6, 1.0, 2.0, 3.0, 4.5]))        # longueurs courtes comprises
            sel = np.zeros(len(P), dtype=bool)
            for k in range(2):
                u = us[k]; uxy = np.hypot(u[0], u[1]); ex, ey = u[0] / uxy, u[1] / uxy
                s = np.clip((P[:, 0] - v[0]) * ex + (P[:, 1] - v[1]) * ey, 0, ls[k] * uxy)
                d = np.hypot(P[:, 0] - (v[0] + s * ex), P[:, 1] - (v[1] + s * ey))
                sel |= d < wide * pitch
            idx = list(np.where(sel)[0])
            if e % 5 == 4 and idx:
                idx.append(idx[0])                       # pad demande deux fois : le dernier doit gagner
                idx.append(len(P) + 7)                   # pad inexistant : doit etre ignore
            f.write("%d %d %.17g %.17g %.17g" % (e, e % 2, v[0], v[1], v[2]))
            f.write(" %d\n" % (e % 3 + 1) if fastfmt else "\n")
            for k in range(2):
                f.write("%.17g %.17g %.17g %.17g\n" % (us[k][0], us[k][1], us[k][2], ls[k]))
            f.write("%d\n" % len(idx))
            for i in idx:
                f.write("%d %d %d\n" % (i, 15 + rng.randint(0, 10), 90 + rng.randint(0, 30)))

if __name__ == "__main__":
    main()
