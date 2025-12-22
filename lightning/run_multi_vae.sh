# Random split (기본값)
python train_multi_vae.py data.split_strategy=random data.valid_ratio=0.1

# Leave-One-Out
python train_multi_vae.py data.split_strategy=leave_one_out

# Temporal User split (유저별 시간)
python train_multi_vae.py data.split_strategy=temporal_user data.temporal_split_ratio=0.8

# Temporal Global split (전역 시간)
python train_multi_vae.py data.split_strategy=temporal_global data.temporal_split_ratio=0.8