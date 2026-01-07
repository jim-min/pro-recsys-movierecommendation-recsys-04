import numpy as np

class EASETrainer:
    def __init__(self, model, train_mat, feat_mats=None):
        self.model = model
        self.train_mat = train_mat
        self.feat_mats = feat_mats

    def train(self):
        self.model.fit(self.train_mat, self.feat_mats)

    def predict(self):
        return self.model.predict(self.train_mat)
