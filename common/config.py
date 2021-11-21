import json
from dataclasses import dataclass


with open('config.json', encoding='utf8') as f:
    config = json.load(f)


@dataclass
class Config:
    """Configuration data."""
    name: str

    # Data
    vocabulary: str
    train_data: str
    test_data: str
    validation_data: str
    use_elmo: bool
    elmo_options: str
    elmo_weights: str

    # Model params
    embed_dim: int
    hidden_dim: int
    dropout: float
    use_gru: bool

    # Training
    learning_rate: float
    n_epochs: int
    early_stopping_epochs: int
    batch_size: int
    batch_shuffle: bool
    max_instances_in_memory: int

    # Saving
    serialization_dir: str
    checkpoints_dir: str

    # CUDA
    use_cuda: bool


configuration = Config(**config)
