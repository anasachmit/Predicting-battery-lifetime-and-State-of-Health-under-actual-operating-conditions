# -*- coding: utf-8 -*-
# =============================================================================
#  FRAMEWORK - DO NOT EDIT.
#  Saves and loads your trained model automatically - regardless of whether
#  you use numpy, scipy, scikit-learn, PyTorch, TensorFlow or anything else.
# =============================================================================
"""
Default behaviour: after --model train, your fitted model OBJECT is pickled
to <work_dir>/model.pkl; --model test (or valid) loads it back. You never write this
file yourself. Pickle covers numpy / scipy / scikit-learn / plain Python
and most PyTorch models out of the box.

Only if your framework cannot be pickled (e.g. some TensorFlow/Keras
models), give your model class these two methods and the framework will
use them instead of pickle:

    def save(self, folder):          # store everything you need in folder
    @classmethod
    def load(cls, folder): ...       # rebuild and return the model
"""
import os
import pickle

STATE_FILE = "model.pkl"


def save_model(model, work_dir):
    os.makedirs(work_dir, exist_ok=True)
    if hasattr(model, "save") and callable(model.save):
        model.save(work_dir)
        open(os.path.join(work_dir, ".custom_io"), "w").write(
            model.__class__.__module__ + ":" + model.__class__.__name__)
        return
    with open(os.path.join(work_dir, STATE_FILE), "wb") as fh:
        pickle.dump(model, fh)


def load_model(work_dir, model_class):
    marker = os.path.join(work_dir, ".custom_io")
    if os.path.exists(marker):
        return model_class.load(work_dir)
    with open(os.path.join(work_dir, STATE_FILE), "rb") as fh:
        return pickle.load(fh)
