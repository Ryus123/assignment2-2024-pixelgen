import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np


from model import Generator, Discriminator
from utils_MHGAN import D_train, G_train, G_double_train, save_models

# Classification models:
from sklearn.linear_model import LogisticRegression


class Calibrator():
    def fit(self, y_pred, y_true):
        pass

    def predict(self, y_pred):
        pass

    @staticmethod
    def validate(y_pred, y_true=None):
        y_pred = np.asarray(y_pred)
        # assert y_pred.ndim == 1
        # assert y_pred.dtype.kind == 'f'
        # assert np.all(0 <= y_pred) and np.all(y_pred <= 1)

        if y_true is not None:
            y_true = np.asarray(y_true)
            # assert y_true.shape == y_pred.shape
            # assert y_true.dtype.kind == 'b'

        return y_pred, y_true

# Classifier 
class Identity(Calibrator):
    def fit(self, y_pred, y_true):
        # assert y_true is not None
        Calibrator.validate(y_pred, y_true)

    def predict(self, y_pred):
        Calibrator.validate(y_pred)
        return y_pred


class Linear(Calibrator):
    def __init__(self):
        self.clf = LogisticRegression()

    def fit(self, y_pred, y_true):
        # assert y_true is not None
        y_pred, y_true = Calibrator.validate(y_pred, y_true)
        self.clf.fit(y_pred[:, None], y_true)

    def predict(self, y_pred):
        y_pred, _ = Calibrator.validate(y_pred)
        y_calib = self.clf.predict_proba(y_pred[:, None])[:, 1]
        return y_calib
    



def calibration_score(y_prob, y_true):
    """
    Function returning the calibration score Z.
    If well calibrated, Z should follow a N(0,1) distribution.
    """
    Z = np.sum(y_true - y_prob) / np.sqrt(np.sum(y_prob * (1.0 - y_prob)))
    return Z

def calibration_diagnostic(pred_df, y_true):
    """
    pred_df is a pandas Dataframe !!! Might need to change
    """
    calibration_df = pred_df.apply(calibration_score, axis=0, args=(y_true,))
    return calibration_df


