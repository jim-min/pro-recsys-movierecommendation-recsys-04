import torch
from torch.utils.data import Dataset

class StaticDataset(Dataset):
    def __init__(self, input_matrix, target_matrix=None):
        """
        Args:
            input_matrix (csr_matrix): 모델에 입력으로 들어갈 데이터 (Train Matrix)
            target_matrix (csr_matrix, optional): 정답 비교용 데이터 (Valid Matrix).
                                                  None 이면 input_matrix를 그대로 반환
        """
        self.input_matrix = input_matrix
        self.target_matrix = target_matrix
    
    def __len__(self):
        # 행(유저)의 개수
        return self.input_matrix.shape[0]
    
    def __getitem__(self, index):
        # csr_matrix에서 해당 유저의 행(row)을 꺼낸다
        # shape: (1, num_items)
        input_row = self.input_matrix[index]
        
        # Dense Array로 변환하고 Tensor로 만든다
        input_tensor = torch.FloatTensor(input_row.toarray())
        
        # 차원 축소 (1, num_items) -> (num_items, )
        # DataLoader가 배치단위로 묶을 때 (Batch, 1, items)가 되는걸 방지
        input_tensor = input_tensor.squeeze()
        
        # target이 있는 경우
        if self.target_matrix is not None:
            target_row = self.target_matrix[index]
            target_tensor = torch.FloatTensor(target_row.toarray()).squeeze()
            return input_tensor, target_tensor
        
        
        return input_tensor
    
        