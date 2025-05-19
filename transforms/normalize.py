from typing import Tuple
import numpy as np

class Normalize:
    def __init__(self,mean: Tuple, std: Tuple) -> None:
        self.mean = mean
        self.std = std

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        shape = matrix.shape
        matrix = matrix / 255.0

        if len(shape) == 2:
            matrix = (matrix - self.mean[0]) / self.std[0]
        elif len(shape) == 3:
            matrix = matrix.transpose(2, 0, 1)
            for i in range(3):
                matrix[i] = (matrix[i] - self.mean[i]) / self.std[i]
            matrix = matrix.transpose(1, 2, 0)
        else:
            raise ValueError(f"Unsupported input shape: {shape}")

        assert matrix.shape == shape
        return matrix
