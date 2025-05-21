from dataclasses import dataclass

@dataclass
class EmbeddingPoint:
    theta: float
    magnitude: float
    x: int
    y: int
    block_idx: int