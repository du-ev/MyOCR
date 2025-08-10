from dataclasses import dataclass

@dataclass
class hparams:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    hidden: int = 256
    dropout: float = 0.3
    optimiser: str = "adamw"
    scheduler: str = "cosine"
    batch: int = 64
    epochs: int = 15
    seed: int = 2025