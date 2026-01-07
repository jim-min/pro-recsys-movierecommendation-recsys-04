"""
BERT4Rec Hyperparameter Tuning with Optuna

Usage:
    # Quick tuning (10 trials)
    python tune_bert4rec_optuna.py --n_trials 10

    # Full tuning (50 trials)
    python tune_bert4rec_optuna.py --n_trials 50 --n_jobs 2

    # Resume study
    python tune_bert4rec_optuna.py --study_name bert4rec_study --resume
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.bert4rec import BERT4Rec
from src.data.bert4rec_data import BERT4RecDataModule

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class OptunaObjective:
    """Optuna objective function for BERT4Rec hyperparameter tuning"""

    def __init__(self, data_dir, n_epochs=50, use_pruning=True):
        self.data_dir = data_dir
        self.n_epochs = n_epochs
        self.use_pruning = use_pruning

    def __call__(self, trial: optuna.Trial):
        """
        Objective function called by Optuna for each trial

        Args:
            trial: Optuna trial object

        Returns:
            float: Validation NDCG@10 score
        """

        # =================================================================
        # 1. HYPERPARAMETER SEARCH SPACE
        # =================================================================

        # Model architecture
        hidden_units = trial.suggest_categorical('hidden_units', [64, 128, 256])
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        max_len = trial.suggest_categorical('max_len', [50, 100, 150, 200])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

        # Training hyperparameters
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])

        # Masking strategy
        random_mask_prob = trial.suggest_float('random_mask_prob', 0.1, 0.3)
        last_item_mask_ratio = trial.suggest_float('last_item_mask_ratio', 0.0, 0.5)

        # Seed (optional - not recommended for production)
        # seed = trial.suggest_int('seed', 0, 10000)

        # =================================================================
        # 2. DATA MODULE
        # =================================================================

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

        # Setup data to get num_items
        datamodule.setup()

        # =================================================================
        # 3. MODEL
        # =================================================================

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

        # =================================================================
        # 4. CALLBACKS
        # =================================================================

        callbacks = [
            # Early stopping (aggressive for faster tuning)
            EarlyStopping(
                monitor='val_ndcg@10',
                patience=5,  # Reduced for faster trials
                mode='max',
                verbose=False,
            ),
            # Checkpoint best model
            ModelCheckpoint(
                monitor='val_ndcg@10',
                mode='max',
                save_top_k=1,
                verbose=False,
            ),
        ]

        # Add pruning callback if enabled
        if self.use_pruning:
            callbacks.append(
                PyTorchLightningPruningCallback(trial, monitor='val_ndcg@10')
            )

        # =================================================================
        # 5. TRAINER
        # =================================================================

        trainer = L.Trainer(
            max_epochs=self.n_epochs,
            accelerator='auto',
            devices=1,  # Single device for tuning
            precision='16-mixed',
            gradient_clip_val=5.0,
            callbacks=callbacks,
            logger=False,  # Disable logging for speed
            enable_progress_bar=False,  # Disable progress bar
            enable_model_summary=False,
            enable_checkpointing=True,
        )

        # =================================================================
        # 6. TRAINING
        # =================================================================

        try:
            trainer.fit(model, datamodule=datamodule)

            # Get best validation score
            best_score = trainer.callback_metrics.get('val_ndcg@10', 0.0)

            # Report intermediate values for pruning
            if isinstance(best_score, torch.Tensor):
                best_score = best_score.item()

            return best_score

        except optuna.TrialPruned:
            # Trial was pruned
            raise
        except Exception as e:
            log.error(f"Trial failed with error: {e}")
            return 0.0  # Return worst score on failure


def tune_hyperparameters(
    data_dir: str,
    n_trials: int = 50,
    n_epochs: int = 50,
    study_name: str = "bert4rec_study",
    storage: str = None,
    n_jobs: int = 1,
    use_pruning: bool = True,
    resume: bool = False,
):
    """
    Run Optuna hyperparameter tuning for BERT4Rec

    Args:
        data_dir: Path to data directory
        n_trials: Number of trials to run
        n_epochs: Max epochs per trial
        study_name: Name of the Optuna study
        storage: Database URL for study storage (None = in-memory)
        n_jobs: Number of parallel jobs
        use_pruning: Whether to use pruning for early stopping trials
        resume: Whether to resume existing study
    """

    # =================================================================
    # 1. CREATE/LOAD STUDY
    # =================================================================

    if storage is None:
        # Use SQLite for persistent storage
        storage = f"sqlite:///{study_name}.db"

    # Create or load study
    if resume:
        log.info(f"Resuming study: {study_name}")
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )
    else:
        log.info(f"Creating new study: {study_name}")
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',  # Maximize NDCG@10
            storage=storage,
            load_if_exists=False,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
            ) if use_pruning else optuna.pruners.NopPruner(),
        )

    # =================================================================
    # 2. RUN OPTIMIZATION
    # =================================================================

    objective = OptunaObjective(
        data_dir=data_dir,
        n_epochs=n_epochs,
        use_pruning=use_pruning,
    )

    log.info(f"Starting optimization with {n_trials} trials...")
    log.info(f"Max epochs per trial: {n_epochs}")
    log.info(f"Parallel jobs: {n_jobs}")
    log.info(f"Pruning enabled: {use_pruning}")

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    # =================================================================
    # 3. RESULTS
    # =================================================================

    log.info("\n" + "="*70)
    log.info("OPTIMIZATION COMPLETE")
    log.info("="*70)

    # Best trial
    best_trial = study.best_trial
    log.info(f"\nBest trial: {best_trial.number}")
    log.info(f"Best NDCG@10: {best_trial.value:.4f}")

    log.info("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        log.info(f"  {key}: {value}")

    # Top 5 trials
    log.info("\n" + "-"*70)
    log.info("Top 5 Trials:")
    log.info("-"*70)

    top_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:5]
    for i, trial in enumerate(top_trials, 1):
        log.info(f"\n{i}. Trial {trial.number}")
        log.info(f"   NDCG@10: {trial.value:.4f}")
        log.info(f"   Params: {trial.params}")

    # Save best config to YAML
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
    }

    config_path = output_dir / f"{study_name}_best_config.yaml"
    with open(config_path, 'w') as f:
        OmegaConf.save(best_config, f)

    log.info(f"\n✅ Best config saved to: {config_path}")

    # Optuna visualization (if available)
    try:
        import optuna.visualization as vis

        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(output_dir / f"{study_name}_history.html")

        # Parameter importance
        fig = vis.plot_param_importances(study)
        fig.write_html(output_dir / f"{study_name}_importance.html")

        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(output_dir / f"{study_name}_parallel.html")

        log.info(f"📊 Visualization saved to: {output_dir}/")

    except ImportError:
        log.warning("plotly not installed. Skipping visualization.")

    return study


def main():
    parser = argparse.ArgumentParser(description='BERT4Rec Optuna Tuning')

    # Data
    parser.add_argument('--data_dir', type=str, default='~/data/train/',
                        help='Path to data directory')

    # Optuna settings
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of optimization trials')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Max epochs per trial')
    parser.add_argument('--study_name', type=str, default='bert4rec_study',
                        help='Name of the Optuna study')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs')
    parser.add_argument('--no_pruning', action='store_true',
                        help='Disable pruning')
    parser.add_argument('--resume', action='store_true',
                        help='Resume existing study')

    args = parser.parse_args()

    # Expand path
    data_dir = os.path.expanduser(args.data_dir)

    # Run tuning
    study = tune_hyperparameters(
        data_dir=data_dir,
        n_trials=args.n_trials,
        n_epochs=args.n_epochs,
        study_name=args.study_name,
        n_jobs=args.n_jobs,
        use_pruning=not args.no_pruning,
        resume=args.resume,
    )

    log.info("\n🎉 Tuning complete!")


if __name__ == '__main__':
    main()
