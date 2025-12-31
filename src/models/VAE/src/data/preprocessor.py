import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        
        # user2idx 역할
        self.user_encoder = LabelEncoder()
        
        # item2idx 역할
        self.item_encoder = LabelEncoder()
    
    # user-item implicit data를 csr-matrix로 변환 전처리
    def load_and_process(self):
        df = pd.read_csv(self.data_path)
        
        # 기존의 고유 ID를 0 ~ N-1 로 변환
        df['user_idx'] = self.user_encoder.fit_transform(df['user'])
        df['item_idx'] = self.item_encoder.fit_transform(df['item'])
        
        self.num_users = len(self.user_encoder.classes_)
        self.num_items = len(self.item_encoder.classes_)
        
        return df
    
    def split_data(self, df : pd.DataFrame, test_ratio=0.01, seed=42):
        """_summary_
        비율이 0.01 이라도 최소 1개의 아이템은 무조건 Valid Set으로 보내는 안전한 분할
        Args:
            df (_type_): _description_
            test_ratio (float, optional): _description_. Defaults to 0.01.
            seed (int, optional): _description_. Defaults to 42.
        """
        np.random.seed(seed)
        
        # 데이터 셔플
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # 각 유저별 아이템 개수 계산
        df['count'] = df.groupby('user_idx')['item_idx'].transform('count')
        
        # Valid Set 개수 계산
        num_valid = np.ceil(df['count'] * test_ratio).astype(int)
        
        # rank 메소드로 그룹내 순번 매기기
        rank = df.groupby('user_idx').cumcount()
        
        is_valid = rank < num_valid
        
        valid_df = df[is_valid].copy()
        train_df = df[~is_valid].copy()
        
        print(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")
        
        # Valid set이 0인 유저가 있는지 여부확인(없어야함)
        valid_user_counts = valid_df.groupby('user_idx').size()
        if len(valid_user_counts) != self.num_users:
            print("Warning: Some users have 0 items in valid set")
        
        # matrix 변환
        train_matrix = self._create_csr_matrix(train_df)
        valid_matrix = self._create_csr_matrix(valid_df)
        
        return train_matrix, valid_matrix
    
    def _create_csr_matrix(self, df):
        row = df['user_idx'].values
        col = df['item_idx'].values
        data = np.ones(len(df))
        
        return csr_matrix((data, (row, col)), shape=(self.num_users, self.num_items))


# if __name__ == "__main__":
#     processor = DataProcessor('./data/train/train_ratings.csv')
#     df = processor.load_and_process()
    
#     train_mat, valid_mat = processor.split_data(df, test_ratio=0.01)
    
#     print(f"Train matrix Shape : {train_mat.shape}")
#     print(f"Valid matrix Shape : {valid_mat.shape}")
    
#     # 첫번째 유저의 아이템 개수 확인
#     u0_train_count = train_mat[25000].sum()
#     u0_valid_count = valid_mat[25000].sum()
#     print(f"User 0 - Train items: {u0_train_count}, Valid items: {u0_valid_count}")