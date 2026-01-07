import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader

from data_utils import read_interactions, encode_ids, build_user_sequences, set_seed
from model import S3RecPretrain
from trainer import S3RecPretrainTrainer


class PretrainDataset(Dataset):
    """
    S3Rec pretrain용 (MIP)
    - 유저별 시퀀스에서 최근 max_len을 가져오고
    - 일부 토큰을 MASK로 바꾸고
    - labels는 masked position만 정답, 나머지는 -100
    """

    def __init__(self, seqs, num_items, max_len=50, mask_prob=0.2, seed=42):
        self.seqs = seqs
        self.num_items = num_items
        self.max_len = max_len
        self.mask_prob = mask_prob

        self.PAD = 0
        self.MASK = num_items + 1

        import numpy as np
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        tokens = [it + 1 for it in seq]  # shift to 1..num_items
        tokens = tokens[-self.max_len:]

        if len(tokens) < self.max_len:
            tokens = [self.PAD] * (self.max_len - len(tokens)) + tokens

        input_ids = tokens[:]
        labels = [-100] * self.max_len

        cand_pos = [i for i, t in enumerate(tokens) if t != self.PAD]
        if len(cand_pos) > 0:
            num_mask = max(1, int(len(cand_pos) * self.mask_prob))
            mask_pos = self.rng.choice(cand_pos, size=num_mask, replace=False).tolist()
        else:
            mask_pos = []

        for p in mask_pos:
            labels[p] = tokens[p]
            input_ids[p] = self.MASK

        pad_mask = [t == self.PAD for t in input_ids]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pad_mask": torch.tensor(pad_mask, dtype=torch.bool),
        }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="/data/ephemeral/home/Seung/data/train/")
    parser.add_argument("--data_file", default="train_ratings.csv")
    parser.add_argument("--output_dir", default="/data/ephemeral/home/Seung/output/S3RecPretrain/")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--mask_prob", type=float, default=0.2)

    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--export_item_emb", default="", help="path to save item embedding pt")

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("🚀 S3Rec Pretrain (MIP)")
    print("=" * 60)

    df = read_interactions(args.data_dir, args.data_file)
    df_enc, u2i, i2u, it2i, i2it = encode_ids(df)
    num_users = len(u2i)
    num_items = len(it2i)
    print(f"Users: {num_users}, Items: {num_items}")

    seqs = build_user_sequences(df_enc, num_users)

    dataset = PretrainDataset(
        seqs=seqs,
        num_items=num_items,
        max_len=args.max_len,
        mask_prob=args.mask_prob,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = S3RecPretrain(
        num_items=num_items,
        max_len=args.max_len,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    trainer = S3RecPretrainTrainer(
        model=model,
        train_loader=loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        use_amp=True,
    )

    best_path = trainer.train(epochs=args.epochs, log_every=200)

    # best 로드 후 item embedding export
    model.load_state_dict(torch.load(best_path, map_location=device))
    model = model.to(device)

    if args.export_item_emb.strip() == "":
        export_path = os.path.join(args.output_dir, "s3rec_item_embedding.pt")
    else:
        export_path = args.export_item_emb.strip()

    model.export_item_embedding(export_path)
    print("✅ Saved S3Rec item embedding:", export_path)


if __name__ == "__main__":
    main()
