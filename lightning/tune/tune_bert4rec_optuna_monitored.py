"""
BERT4Rec Optuna Tuning with Enhanced Monitoring

추가 기능:
- 실시간 progress tracking
- Trial별 상세 로그
- Intermediate value reporting
- Best trial auto-save
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parent.parent))

from src.models.bert4rec import BERT4Rec
from src.data.bert4rec_data import BERT4RecDataModule

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


class MetricsCallback(Callback):
    """Custom callback to report metrics to Optuna"""

    def __init__(self, trial: optuna.Trial):
        super().__init__()
        self.trial = trial

    def on_validation_epoch_end(self, trainer, pl_module):
        """Report metrics after each validation epoch"""
        # Get current epoch
        current_epoch = trainer.current_epoch

        # Get validation NDCG
        metrics = trainer.callback_metrics
        val_ndcg = metrics.get('val_ndcg@10', 0.0)

        if isinstance(val_ndcg, torch.Tensor):
            val_ndcg = val_ndcg.item()

        # Report intermediate value to Optuna
        self.trial.report(val_ndcg, current_epoch)

        # Log to console
        log.info(f"Trial {self.trial.number} | Epoch {current_epoch} | NDCG@10: {val_ndcg:.4f}")


class OptunaObjective:
    """Enhanced Optuna objective with monitoring"""

    def __init__(self, data_dir, n_epochs=50, use_pruning=True):
        self.data_dir = data_dir
        self.n_epochs = n_epochs
        self.use_pruning = use_pruning
        self.best_score = 0.0
        self.trial_count = 0

    def __call__(self, trial: optuna.Trial):
        """Objective function with enhanced monitoring"""

        self.trial_count += 1

        # Print trial header
        log.info("=" * 80)
        log.info(f"TRIAL {trial.number} START (Total: {self.trial_count})")
        log.info("=" * 80)

        # Hyperparameters
        hidden_units = trial.suggest_categorical('hidden_units', [64, 128, 256])
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        max_len = trial.suggest_categorical('max_len', [50, 100, 150, 200])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])

        random_mask_prob = trial.suggest_float('random_mask_prob', 0.1, 0.3)
        last_item_mask_ratio = trial.suggest_float('last_item_mask_ratio', 0.0, 0.5)

        # Print hyperparameters
        log.info(f"\nTrial {trial.number} Hyperparameters:")
        log.info(f"  Model: hidden={hidden_units}, heads={num_heads}, layers={num_layers}, max_len={max_len}")
        log.info(f"  Training: lr={lr:.6f}, weight_decay={weight_decay:.4f}, batch={batch_size}")
        log.info(f"  Masking: random={random_mask_prob:.2f}, last_item={last_item_mask_ratio:.2f}")
        log.info("")

        # DataModule
        datamodule = BERT4RecDataModule(
            data_dir=self.data_dir,
            data_file="train_ratings.csv",
            batch_size=batch_size,
            max_len=max_len,
            random_mask_prob=random_mask_prob,
            last_item_mask_ratio=last_item_mask_ratio,
            min_interactions=3,
            seed=42,
            num_workers=4,
            use_full_data=False,
        )
        datamodule.setup()

        # Model
        model = BERT4Rec(
            num_items=datamodule.num_items,
            hidden_units=hidden_units,
            num_heads=num_heads,
            num_layers=num_layers,
            max_len=max_len,
            dropout_rate=dropout_rate,
            random_mask_prob=random_mask_prob,
            last_item_mask_ratio=last_item_mask_ratio,
            lr=lr,
            weight_decay=weight_decay,
            share_embeddings=True,
        )

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_ndcg@10',
                patience=5,
                mode='max',
                verbose=True,
            ),
            ModelCheckpoint(
                monitor='val_ndcg@10',
                mode='max',
                save_top_k=1,
                verbose=False,
            ),
            MetricsCallback(trial),  # Custom callback for Optuna reporting
        ]

        if self.use_pruning:
            callbacks.append(
                PyTorchLightningPruningCallback(trial, monitor='val_ndcg@10')
            )

        # Trainer
        trainer = L.Trainer(
            max_epochs=self.n_epochs,
            accelerator='auto',
            devices=1,
            precision='16-mixed',
            gradient_clip_val=5.0,
            callbacks=callbacks,
            logger=False,
            enable_progress_bar=True,  # Enable for monitoring
            enable_model_summary=False,
            enable_checkpointing=True,
        )

        # Training
        try:
            start_time = datetime.now()

            trainer.fit(model, datamodule=datamodule)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Get best score
            best_score = trainer.callback_metrics.get('val_ndcg@10', 0.0)
            if isinstance(best_score, torch.Tensor):
                best_score = best_score.item()

            # Update global best
            if best_score > self.best_score:
                self.best_score = best_score
                log.info(f"\n🎉 NEW BEST SCORE: {best_score:.4f}")

            # Print trial summary
            log.info("=" * 80)
            log.info(f"TRIAL {trial.number} COMPLETE")
            log.info(f"  Score: {best_score:.4f}")
            log.info(f"  Duration: {duration/60:.1f} minutes")
            log.info(f"  Current Best: {self.best_score:.4f}")
            log.info("=" * 80)
            log.info("")

            return best_score

        except optuna.TrialPruned:
            log.info(f"Trial {trial.number} was pruned")
            raise
        except Exception as e:
            log.error(f"Trial {trial.number} failed: {e}")
            return 0.0


def tune_with_monitoring(
    data_dir: str,
    n_trials: int = 50,
    n_epochs: int = 50,
    study_name: str = "bert4rec_monitored",
    n_jobs: int = 1,
    use_pruning: bool = True,
    resume: bool = False,
):
    """
    Run Optuna tuning with enhanced monitoring
    """

    storage = f"sqlite:///{study_name}.db"

    # Create/load study
    if resume:
        log.info(f"Resuming study: {study_name}")
        study = optuna.load_study(study_name=study_name, storage=storage)
    else:
        log.info(f"Creating new study: {study_name}")
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            storage=storage,
            load_if_exists=False,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
            ) if use_pruning else optuna.pruners.NopPruner(),
        )

    # Print study info
    log.info("\n" + "=" * 80)
    log.info("OPTUNA STUDY CONFIGURATION")
    log.info("=" * 80)
    log.info(f"Study name: {study_name}")
    log.info(f"Storage: {storage}")
    log.info(f"Number of trials: {n_trials}")
    log.info(f"Max epochs per trial: {n_epochs}")
    log.info(f"Parallel jobs: {n_jobs}")
    log.info(f"Pruning: {use_pruning}")
    log.info("=" * 80)
    log.info("")

    # Print monitoring instructions
    log.info("💡 MONITORING TIPS:")
    log.info("  1. Open Optuna Dashboard in another terminal:")
    log.info(f"     $ optuna-dashboard {storage}")
    log.info("     Then open http://127.0.0.1:8080 in browser")
    log.info("")
    log.info("  2. Check database:")
    log.info(f"     $ sqlite3 {study_name}.db 'SELECT number, value FROM trials ORDER BY value DESC LIMIT 5;'")
    log.info("")
    log.info("=" * 80)
    log.info("")

    # Objective
    objective = OptunaObjective(
        data_dir=data_dir,
        n_epochs=n_epochs,
        use_pruning=use_pruning,
    )

    # Optimize with progress bar
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
        callbacks=[lambda study, trial: print_trial_callback(study, trial)],
    )

    # Results
    log.info("\n" + "=" * 80)
    log.info("OPTIMIZATION COMPLETE")
    log.info("=" * 80)

    best_trial = study.best_trial
    log.info(f"\n🏆 Best trial: {best_trial.number}")
    log.info(f"🎯 Best NDCG@10: {best_trial.value:.4f}")

    log.info("\n📊 Best hyperparameters:")
    for key, value in best_trial.params.items():
        log.info(f"  {key:25s}: {value}")

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    best_config = {
        'model': {
            'hidden_units': best_trial.params['hidden_units'],
            'num_heads': best_trial.params['num_heads'],
            'num_layers': best_trial.params['num_layers'],
            'max_len': best_trial.params['max_len'],
            'dropout_rate': best_trial.params['dropout_rate'],
            'random_mask_prob': best_trial.params['random_mask_prob'],
            'last_item_mask_ratio': best_trial.params['last_item_mask_ratio'],
        },
        'training': {
            'lr': best_trial.params['lr'],
            'weight_decay': best_trial.params['weight_decay'],
        },
        'data': {
            'batch_size': best_trial.params['batch_size'],
        },
        'best_score': best_trial.value,
        'trial_number': best_trial.number,
    }

    config_path = output_dir / f"{study_name}_best_config.yaml"
    with open(config_path, 'w') as f:
        OmegaConf.save(best_config, f)

    log.info(f"\n✅ Best config saved to: {config_path}")

    # Print study statistics
    log.info("\n" + "=" * 80)
    log.info("STUDY STATISTICS")
    log.info("=" * 80)
    log.info(f"Total trials: {len(study.trials)}")
    log.info(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    log.info(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    log.info(f"Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

    return study


def print_trial_callback(study, trial):
    """Callback to print after each trial"""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        log.info(f"\n✓ Trial {trial.number} completed with score: {trial.value:.4f}")
        log.info(f"  Current best: {study.best_value:.4f} (Trial {study.best_trial.number})")


def main():
    parser = argparse.ArgumentParser(description='BERT4Rec Optuna Tuning with Monitoring')

    parser.add_argument('--data_dir', type=str, default='~/data/train/')
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--study_name', type=str, default='bert4rec_monitored')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--no_pruning', action='store_true')
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)

    study = tune_with_monitoring(
        data_dir=data_dir,
        n_trials=args.n_trials,
        n_epochs=args.n_epochs,
        study_name=args.study_name,
        n_jobs=args.n_jobs,
        use_pruning=not args.no_pruning,
        resume=args.resume,
    )

    log.info("\n🎉 Tuning complete!")
    log.info(f"📊 View results: optuna-dashboard sqlite:///{args.study_name}.db")


if __name__ == '__main__':
    main()
