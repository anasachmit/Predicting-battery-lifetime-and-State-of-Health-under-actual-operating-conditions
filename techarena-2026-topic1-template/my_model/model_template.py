# -*- coding: utf-8 -*-
# =============================================================================
#
#   >>> THIS IS YOUR FILE - IMPLEMENT YOUR MODEL HERE <<<
#
#   You implement exactly TWO methods:  fit()  and  predict_soh().
#   Additional helper functions and classes for data preprocessing and
#   feature extraction are welcome, but the framework will only call these two.
#   Everything else (loading data, saving/loading your model, targets,
#   output files, stopping at 70%) is handled by the framework.
#
#   A complete, working reference is next to this file: model_example.py.
#   To run YOUR model instead of the example, switch one line in
#   my_model/__init__.py.
#
# =============================================================================
"""What you get and what you must return:

fit(cells)
    cells is a list of Cell objects, one per training cell:
        cell.temperature_degC   -> 25 / 35 / 45 / 55
        cell.c_rate             -> 0.5 / 1.0
        cell.soh                -> DataFrame [cycle_number, soh_percent]
        cell.time_series()      -> full raw signals if you want features
                                   (voltage, current, temperature, ...)
    Learn whatever you need and keep it on self.* - the framework
    saves your whole object automatically after training.

predict_soh(temperature_degC, c_rate)
    Return a 1D array of SOH values in % for cycles 1, 2, 3, ... for the
    requested operating point (anywhere in 25-55 degC x 0.5-1.0 C).
    Returning 12000 values is always safe - the framework picks the
    cycles it needs and builds output.csv for you.
"""
import numpy as np


class MyModel:

    def __init__(self):
        # TODO: your hyperparameters / components go here
        pass

    def fit(self, cells):
        # TODO: train on the provided cells. BOTH input styles are supported:
        #
        # (A) Label-curve models - use only cycle number and SOH:
        #       for cell in cells:
        #           lab = cell.soh.dropna()      # cycle_number, soh_percent
        #           # fit fade curves, trajectory models, ...
        #
        # (B) Signal-based models - use the raw time series as input.
        #     Everything is in cell.time_series(); e.g. per cycle:
        #
        #       ts = cell.time_series()
        #       cyc = ts[ts["cycle_number"] == 100]
        #
        #       # voltage / temperature profile of that cycle:
        #       chg = cyc[cyc["step_type"].isin(["cccv_charge", "cc_charge"])]
        #       v_profile = chg["voltage_V"].to_numpy()
        #       t_profile = chg["temperature_C"].to_numpy()
        #
        #       # ICA (incremental capacity, dQ/dV) of the charge:
        #       dqdv = np.gradient(chg["step_capacity_Ah"].to_numpy(),
        #                          chg["voltage_V"].to_numpy())
        #
        #       # DC resistance at discharge start (|dV/dI| at the
        #       # rest -> cc_discharge transition):
        #       i = cyc.index[cyc["step_type"] == "cc_discharge"][0]
        #       R = abs((cyc["voltage_V"][i] - cyc["voltage_V"][i - 1])
        #               / (cyc["current_A"][i] - cyc["current_A"][i - 1]))
        #
        #     Feed such features into sequence encoders, networks
        #     pretrained on open data, physics-informed models, ...
        #
        # Either way: everything you learn must end up on self.*, because
        # at prediction time the ONLY information about a target is
        # (temperature, C-rate) - hidden cells have no measured data by
        # definition. Pre-train on open data OFFLINE in your own scripts,
        # ship the weights inside this folder, and load them here.
        raise NotImplementedError("implement fit() in my_model/model_template.py")

    def predict_soh(self, temperature_degC, c_rate):
        # TODO: return SOH % for cycles 1, 2, 3, ... as a 1D array.
        #
        #   n = np.arange(1, 12001)
        #   soh = <your model>(n, temperature_degC, c_rate)
        #   return soh
        raise NotImplementedError("implement predict_soh() in my_model/model_template.py")

    # -------------------------------------------------------------------------
    # OPTIONAL - only needed if your framework cannot be pickled (e.g. some
    # TensorFlow models). Otherwise DELETE nothing and IGNORE this:
    #
    # def save(self, folder): ...
    # @classmethod
    # def load(cls, folder): ...
    # -------------------------------------------------------------------------
