"""
Quick Tuning Script for BERT4Rec
빠른 하이퍼파라미터 탐색을 위한 스크립트

Usage:
    # 빠른 튜닝 (10 trials, 20 epochs)
    python quick_tune.py

    # 중간 튜닝 (30 trials, 50 epochs)
    python quick_tune.py --mode medium

    # 전체 튜닝 (100 trials, 100 epochs)
    python quick_tune.py --mode full
"""

import argparse
from tune_bert4rec_optuna import tune_hyperparameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "quick", "medium", "full"],
        help="Tuning mode",
    )
    parser.add_argument("--data_dir", type=str, default="~/data/train/")

    args = parser.parse_args()

    # Mode configurations
    configs = {
        "test": {
            "n_trials": 2,
            "n_epochs": 2,
            "description": "Test scripts (1 trial, 1 epochs) - ~2-3 minutes",
        },
        "quick": {
            "n_trials": 10,
            "n_epochs": 20,
            "description": "Quick tuning (10 trials, 20 epochs) - ~2-3 hours",
        },
        "medium": {
            "n_trials": 30,
            "n_epochs": 50,
            "description": "Medium tuning (30 trials, 50 epochs) - ~8-12 hours",
        },
        "full": {
            "n_trials": 100,
            "n_epochs": 100,
            "description": "Full tuning (100 trials, 100 epochs) - ~1-2 days",
        },
    }

    config = configs[args.mode]

    print("=" * 70)
    print(f"BERT4Rec Hyperparameter Tuning - {args.mode.upper()} MODE")
    print("=" * 70)
    print(f"\n{config['description']}")
    print(f"\nSettings:")
    print(f"  - Trials: {config['n_trials']}")
    print(f"  - Max epochs per trial: {config['n_epochs']}")
    print(f"  - Data directory: {args.data_dir}")
    print("\n" + "=" * 70 + "\n")

    # Run tuning
    study = tune_hyperparameters(
        data_dir=args.data_dir,
        n_trials=config["n_trials"],
        n_epochs=config["n_epochs"],
        study_name=f"bert4rec_{args.mode}",
        n_jobs=1,
        use_pruning=True,
        resume=False,
    )

    print("\n✅ Tuning complete!")
    print(f"📊 Results saved to: tune/results/bert4rec_{args.mode}_best_config.yaml")


if __name__ == "__main__":
    main()
