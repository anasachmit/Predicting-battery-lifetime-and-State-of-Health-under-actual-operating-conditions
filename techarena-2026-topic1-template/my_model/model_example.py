# -*- coding: utf-8 -*-
# =============================================================================
#  WORKING REFERENCE EXAMPLE - runs out of the box, defines score = 1.0.
#  Read it, run it, then build something better in model_template.py.
# =============================================================================
"""A deliberately simple physics-inspired baseline:

fit:      per cell, fit a power-law fade  soh(n) = a - b * n^0.8  to the
          labels; then model  ln(b)  as a linear function of 1000/T_kelvin
          (Arrhenius-style) and ln(C-rate) across the training cells.
predict:  rebuild b for the requested (T, C) - including points between
          the training conditions - and roll the fade curve forward until
          it crosses 70%.
"""
import numpy as np

FADE_P = 0.8


class ExampleModel:

    def fit(self, cells):
        rows = []
        for cell in cells:
            d = cell.soh.dropna()
            n = d["cycle_number"].to_numpy(float)
            y = d["soh_percent"].to_numpy(float)
            if len(n) < 10:
                continue
            X = np.column_stack([np.ones_like(n), n ** FADE_P])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            a, b = float(coef[0]), float(-coef[1])
            if np.isfinite(a) and np.isfinite(b) and b > 0:
                rows.append((a, b, cell.temperature_degC, cell.c_rate))
                print(f"  (example) {cell.cell_id}: a={a:.2f} b={b:.4f}")
        if not rows:
            raise RuntimeError("no usable training cells")
        A = np.array(rows)
        self.a_ = float(A[:, 0].mean())
        Tk = A[:, 2] + 273.15
        X = np.column_stack([np.ones(len(A)), 1000.0 / Tk, np.log(A[:, 3])])
        y = np.log(A[:, 1])
        self.w_ = (np.linalg.lstsq(X, y, rcond=None)[0] if len(A) >= 3
                   else np.array([float(y.mean()), 0.0, 0.0]))

    def predict_soh(self, temperature_degC, c_rate):
        Tk = float(temperature_degC) + 273.15
        b = float(np.exp(self.w_[0] + self.w_[1] * 1000.0 / Tk
                         + self.w_[2] * np.log(float(c_rate))))
        n = np.arange(1, 12001, dtype=float)
        return np.clip(self.a_ - b * n ** FADE_P, 0.5, 119.9)
