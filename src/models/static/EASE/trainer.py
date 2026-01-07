import numpy as np


class EASETrainer:
    def __init__(self, model, train_mat):
        self.model = model
        self.train_mat = train_mat

    def train(self):
        self.model.fit(self.train_mat)

    def predict(self):
        return self.model.predict(self.train_mat)
