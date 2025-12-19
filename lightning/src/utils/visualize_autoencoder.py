import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import logging
import torch
from src.models.auto_encoder import AutoEncoder


def find_best_checkpoint(checkpoint_dir="../saved/logs/checkpoints"):
    """
    체크포인트 디렉토리에서 가장 최근 또는 최적의 체크포인트를 찾습니다.

    Args:
        checkpoint_dir: 체크포인트가 저장된 디렉토리

    Returns:
        checkpoint_path: 체크포인트 파일 경로
    """
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))

    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    # 가장 최근 파일 선택 (수정 시간 기준)
    latest_ckpt = max(ckpt_files, key=os.path.getmtime)
    logging.info(f"Found checkpoint: {latest_ckpt}")

    return latest_ckpt


def load_model_from_checkpoint(checkpoint_path):
    """
    체크포인트에서 모델을 로드합니다.

    Args:
        checkpoint_path: 체크포인트 파일 경로

    Returns:
        model: 로드된 AutoEncoder 모델
    """
    logging.info(f"Loading model from checkpoint: {checkpoint_path}")

    # Lightning 모델 로드
    model = AutoEncoder.load_from_checkpoint(checkpoint_path)
    model.eval()

    logging.info("Model loaded successfully")
    return model


def visualize_latent_space(autoencoder, datamodule, target_labels, use_train=True):
    """
    Test 또는 Train sample의 latent space 좌표를 추출하고 plot

    Args:
        autoencoder: AutoEncoder 모델
        datamodule: MNISTDataModule 인스턴스
        target_labels: 타겟 레이블 리스트
        use_train: True면 train_dataloader, False면 test_dataloader 사용
    """
    # 모델을 evaluation 모드로 설정
    autoencoder.eval()

    # 모델이 있는 디바이스 확인
    device = next(autoencoder.parameters()).device
    logging.info(f"Model is on device: {device}")

    # 데이터로더 선택
    dataloader = (
        datamodule.train_dataloader() if use_train else datamodule.test_dataloader()
    )

    # 모든 데이터 가져오기
    latent_coords = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.view(x.size(0), -1)  # flatten

            # 입력 텐서를 모델과 같은 디바이스로 이동
            x = x.to(device)

            # encoder를 통과하여 latent space 좌표 얻기
            z = autoencoder.encoder(x)

            latent_coords.append(z.cpu().numpy())
            labels.append(y.cpu().numpy())

    # 리스트를 numpy array로 변환
    latent_coords = np.concatenate(latent_coords, axis=0)
    labels = np.concatenate(labels, axis=0)

    logging.info(f"Latent coordinates shape: {latent_coords.shape}")
    logging.info(f"Labels shape: {labels.shape}")

    # Scatter plot 생성
    plt.figure(figsize=(10, 8))

    for label in target_labels:
        mask = labels == label
        plt.scatter(
            latent_coords[mask, 0],
            latent_coords[mask, 1],
            label=f"Label {label}",
            alpha=0.6,
            s=30,
        )

    # 박스 그리기 ([-10, 10] 범위)
    box_x = [-10, 10, 10, -10, -10]  # 마지막은 시작점으로 닫기
    box_y = [-10, -10, 10, 10, -10]
    plt.plot(box_x, box_y, "r-", linewidth=2)  # 빨간색 실선

    plt.xlabel("Latent Dimension 1", fontsize=12)
    plt.ylabel("Latent Dimension 2", fontsize=12)
    dataset_type = "Train" if use_train else "Test"
    plt.title(
        f"Latent Space Visualization of {dataset_type} Samples",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.axis("equal")
    plt.show()

    # 통계 정보 출력
    print(f"\n=== Latent Space Statistics ===")
    for label in target_labels:
        mask = labels == label
        coords = latent_coords[mask]
        print(f"\nLabel {label}:")
        print(f"  Count: {coords.shape[0]}")
        print(
            f"  Dim 1 - Mean: {coords[:, 0].mean():.4f}, Std: {coords[:, 0].std():.4f}"
        )
        print(
            f"  Dim 2 - Mean: {coords[:, 1].mean():.4f}, Std: {coords[:, 1].std():.4f}"
        )


def generate_from_latent(model, latent_points, figsize=(12, 4)):
    """
    특정 latent 포인트들로부터 이미지 생성

    Args:
        model: AutoEncoder 모델
        latent_points: (n, 2) shape의 numpy array 또는 리스트
        figsize: figure 크기

    Returns:
        generated_images: 생성된 이미지들 (numpy array)
    """
    model.eval()

    # 모델이 있는 디바이스 확인
    device = next(model.parameters()).device

    # numpy array로 변환
    if isinstance(latent_points, list):
        latent_points = np.array(latent_points)

    # tensor로 변환하고 모델과 같은 디바이스로 이동
    latent_tensor = torch.FloatTensor(latent_points).to(device)

    # decoder를 통해 이미지 생성
    with torch.no_grad():
        generated_images = model.decoder(latent_tensor)
        generated_images = generated_images.cpu().numpy()

    # 이미지를 28x28로 reshape
    generated_images = generated_images.reshape(-1, 28, 28)

    # 시각화
    n_images = len(latent_points)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)

    if n_images == 1:
        axes = [axes]

    for i, (img, point) in enumerate(zip(generated_images, latent_points)):
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"({point[0]:.2f}, {point[1]:.2f})", fontsize=10)
        axes[i].axis("off")

    plt.suptitle("Generated Images from Latent Points", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return generated_images


def generate_latent_grid(autoencoder, n_samples=20, latent_range=10):
    """
    임의의 latent space 포인트로부터 이미지 생성 (Grid 시각화)

    Args:
        autoencoder: LitAutoEncoder 모델
        n_samples: 한 축당 샘플 개수
        latent_range: latent space 범위 (-latent_range ~ +latent_range)
    """
    autoencoder.eval()

    # 모델이 있는 디바이스 확인
    device = next(autoencoder.parameters()).device

    # latent space에서 grid 포인트 생성
    x = np.linspace(-latent_range, latent_range, n_samples)
    y = np.linspace(-latent_range, latent_range, n_samples)
    xx, yy = np.meshgrid(x, y)

    # 그리드 포인트들을 (n_samples^2, 2) 형태로 변환
    latent_points = np.stack([xx.flatten(), np.flip(yy).flatten()], axis=1)

    # latent 포인트를 tensor로 변환하고 모델과 같은 디바이스로 이동
    latent_tensor = torch.FloatTensor(latent_points).to(device)

    # decoder를 통해 이미지 생성
    with torch.no_grad():
        generated_images = autoencoder.decoder(latent_tensor)
        generated_images = generated_images.cpu().numpy()

    # 이미지를 28x28로 reshape
    generated_images = generated_images.reshape(-1, 28, 28)

    # 생성된 이미지를 grid로 시각화
    fig = plt.figure(figsize=(8, 8))

    for i in range(n_samples * n_samples):
        ax = plt.subplot(n_samples, n_samples, i + 1)
        plt.imshow(generated_images[i], cmap="gray")
        plt.axis("off")

    plt.suptitle(
        f"Generated Images from Latent Space Grid (Range: [{-latent_range}, {latent_range}])",
        y=1,
    )
    plt.tight_layout(pad=0.3)
    plt.show()

    logging.info(f"Generated {n_samples * n_samples} images from latent space grid")


def visualize_sample_data(dataloader, n_samples=10):
    """
    데이터로더에서 샘플 이미지들을 시각화

    Args:
        dataloader: 데이터로더
        n_samples: 표시할 샘플 개수
    """
    (X, y) = next(iter(dataloader))

    plt.figure(figsize=(7, 4))
    for i in range(min(n_samples, len(X))):
        plt.subplot(2, 5, i + 1)
        plt.title(y[i].item())
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i, 0], cmap="gray", interpolation="none")
    plt.tight_layout()
    plt.show()
