import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

class Ease:
    def __init__(self, reg_lambda=500):
        """_summary_

        Args:
            reg_lambda (int, optional): 정규화 강도
        """
        self.reg_lambda = reg_lambda
        self.weight = None
        
    def fit(self, X: csr_matrix):
        """
        Closed-form Solution 계산
        Args:
            X (csr_matrix): 유저-아이템 상호작용 행렬 (Users X Items)
        """
        
        # csr_matrix 상태로 X.T @ X = G 계산
        if sp.issparse(X):
            G_sparse = X.transpose().dot(X)
            G_dense = G_sparse.toarray()
            G = torch.from_numpy(G_dense).float()
        else:
            X_torch = torch.from_numpy(X).float()
            G = X_torch.t() @ X_torch
    
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        G = G.to(device)
        
        # 대각 성분에 lambda 추가
        # G += lambda * I
        diag_indices = torch.arange(G.shape[0], device=device)
        G[diag_indices, diag_indices] += self.reg_lambda
        
        # 역행렬 계산 (P = G^-1)
        P = torch.linalg.inv(G)
        
        # 가중지 행렬 B 계산 (제약 조건 diag(B) = 0)
        # B_ij = -P_ij / P_jj
        diag_P = torch.diag(P)
        self.weight = P / (-diag_P.view(1, -1))
        
        # 대각 성분 0으로 만들기
        self.weight[diag_indices, diag_indices] = 0
        
        self.weight = self.weight.to(device)
        print("EASE Training Finished")
    
    def predict(self, X: csr_matrix, k=10, remove_seen=True):
        """_summary_

        Args:
            X (scipy.sparse.csr_matrix): User X Items
            k (int, optional): Top-K 추천 개수
            remove_seen (bool, optional): 이미 본 아이템 추천에서 제거여부
        """
        
        device = self.weight.device
        
        # CSR -> torch.Tensor로 변환
        if sp.issparse(X):
            X = torch.from_numpy(X.toarray()).float().to(device)
        # X가 torch.Tensor
        else:
            X = X.float().to(device)
        
        # Users X Items
        scores = X @ self.weight
        
        if remove_seen:
            scores[X.bool()] = -float('inf')
        
        return scores