"""Shared test fixtures for pytest"""
import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from src.models.bert4rec import BERT4Rec


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary test data directory with sample files"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create sample ratings file
    create_sample_ratings(data_dir / "train_ratings.csv")

    # Create sample metadata files
    create_sample_genres(data_dir / "genres.tsv")
    create_sample_directors(data_dir / "directors.tsv")
    create_sample_writers(data_dir / "writers.tsv")
    create_sample_titles(data_dir / "titles.tsv")
    create_sample_years(data_dir / "years.tsv")

    return str(data_dir)


@pytest.fixture
def sample_config():
    """Sample minimal BERT4Rec configuration for testing"""
    return {
        "num_items": 50,
        "hidden_units": 32,
        "num_heads": 2,
        "num_layers": 2,
        "max_len": 10,
        "dropout_rate": 0.1,
        "random_mask_prob": 0.15,
        "last_item_mask_ratio": 0.0,  # Disable last item masking for basic tests
        "lr": 0.001,
        "weight_decay": 0.0,
        "share_embeddings": True,
        # Metadata parameters (minimal)
        "num_genres": 5,
        "num_directors": 10,
        "num_writers": 10,
        "title_embedding_dim": 0,  # No title embeddings for basic tests
        "use_genre_emb": True,
        "use_director_emb": True,
        "use_writer_emb": True,
        "use_title_emb": False,
        "metadata_fusion": "concat",
        "metadata_dropout": 0.1,
    }


@pytest.fixture
def sample_config_no_metadata():
    """Sample BERT4Rec configuration without metadata for testing"""
    return {
        "num_items": 50,
        "hidden_units": 32,
        "num_heads": 2,
        "num_layers": 2,
        "max_len": 10,
        "dropout_rate": 0.1,
        "random_mask_prob": 0.15,
        "last_item_mask_ratio": 0.0,  # Disable last item masking for basic tests
        "lr": 0.001,
        "weight_decay": 0.0,
        "share_embeddings": True,
        # Metadata disabled
        "num_genres": 1,
        "num_directors": 1,
        "num_writers": 1,
        "title_embedding_dim": 0,
        "use_genre_emb": False,
        "use_director_emb": False,
        "use_writer_emb": False,
        "use_title_emb": False,
        "metadata_fusion": "concat",
        "metadata_dropout": 0.1,
    }


@pytest.fixture
def bert4rec_model(sample_config):
    """Create BERT4Rec model instance for testing (with metadata)"""
    return BERT4Rec(**sample_config)


@pytest.fixture
def bert4rec_model_no_metadata(sample_config_no_metadata):
    """Create BERT4Rec model instance for testing (without metadata)"""
    return BERT4Rec(**sample_config_no_metadata)


@pytest.fixture
def sample_batch():
    """Sample batch data (sequences, labels)"""
    batch_size = 4
    seq_len = 10

    sequences = torch.randint(1, 51, (batch_size, seq_len))
    labels = torch.randint(0, 51, (batch_size, seq_len))

    return sequences, labels


@pytest.fixture
def sample_metadata():
    """Sample metadata dictionary"""
    batch_size = 4
    seq_len = 10
    max_genres = 3
    max_writers = 2

    metadata = {
        "genres": torch.randint(0, 5, (batch_size, seq_len, max_genres)),
        "directors": torch.randint(0, 10, (batch_size, seq_len)),
        "writers": torch.randint(0, 10, (batch_size, seq_len, max_writers)),
    }

    return metadata


# Helper functions to create sample data files


def create_sample_ratings(path):
    """Create sample ratings CSV file"""
    data = {
        "user": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5],
        "item": [
            10,
            20,
            30,
            40,
            15,
            25,
            35,
            10,
            20,
            30,
            40,
            12,
            22,
            32,
            11,
            21,
            31,
            41,
            50,
        ],
        "time": [
            1000,
            2000,
            3000,
            4000,
            1500,
            2500,
            3500,
            1100,
            2100,
            3100,
            4100,
            1200,
            2200,
            3200,
            1300,
            2300,
            3300,
            4300,
            5300,
        ],
    }
    pd.DataFrame(data).to_csv(path, index=False)


def create_sample_genres(path):
    """Create sample genres TSV file"""
    data = {
        "item": [10, 10, 20, 20, 30, 30, 40, 40, 50],
        "genre": [
            "Action",
            "Drama",
            "Comedy",
            "Romance",
            "Action",
            "Sci-Fi",
            "Drama",
            "Thriller",
            "Comedy",
        ],
    }
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)


def create_sample_directors(path):
    """Create sample directors TSV file"""
    data = {
        "item": [10, 20, 30, 40, 50],
        "director": ["nm001", "nm002", "nm003", "nm004", "nm005"],
    }
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)


def create_sample_writers(path):
    """Create sample writers TSV file"""
    data = {
        "item": [10, 10, 20, 30, 30, 40, 50, 50],
        "writer": ["nm101", "nm102", "nm103", "nm104", "nm105", "nm106", "nm107", "nm108"],
    }
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)


def create_sample_titles(path):
    """Create sample titles TSV file"""
    data = {
        "item": [10, 20, 30, 40, 50],
        "title": [
            "Movie A (2020)",
            "Movie B (2021)",
            "Movie C (2019)",
            "Movie D (2022)",
            "Movie E (2023)",
        ],
    }
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)


def create_sample_years(path):
    """Create sample years TSV file"""
    data = {"item": [10, 20, 30, 40, 50], "year": [2020, 2021, 2019, 2022, 2023]}
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)
