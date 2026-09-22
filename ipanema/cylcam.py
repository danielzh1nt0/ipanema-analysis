"""Curved (cylindrical) camera for Veo's panorama: a fixed camera on a mast; a pitch point maps to the picture by its
horizontal direction (u) and the tangent of its angle below the horizon (v), after a small tilt/roll of the camera.
Pitch: 106 x 64, x along the pitch, y toward the camera side (0 = far touchline)."""
import numpy as np, cv2, sys
sys.path.insert(0, "/home/claude/push")
from ipanema.calcheck import line_mask, pitch_segments

NAMES = ["cx", "cy", "h", "yaw", "fu", "fv", "u0", "v0", "tilt", "roll"]

def project(params, P):
    cx, cy, h, yaw, fu, fv, u0, v0, tilt, roll = params
    d = np.column_stack([P[:, 0] - cx, P[:, 1] - cy, np.full(len(P), h)])          # z down: ground is h below the camera
    ct, st, cr, sr = np.cos(tilt), np.sin(tilt), np.cos(roll), np.sin(roll)
    Rt = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]]); Rr = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    d = d @ (Rr @ Rt).T
    az = np.arctan2(d[:, 1], d[:, 0]) - yaw; az = (az + np.pi) % (2 * np.pi) - np.pi
    rho = np.hypot(d[:, 0], d[:, 1]); tb = d[:, 2] / np.maximum(rho, 1e-6)
    return np.column_stack([u0 + fu * az, v0 + fv * tb])

def model_pts(L=106.0, W=64.0, step=0.4):
    out = []
    for a, b in pitch_segments(L, W):
        a, b = np.array(a, float), np.array(b, float); n = max(2, int(np.linalg.norm(b - a) / step)); out.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    return np.vstack(out)
