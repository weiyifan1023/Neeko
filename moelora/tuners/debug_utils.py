import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

class DebugUtils:
    def __init__(self, pic_dir: str):
        self.pic_dir = pic_dir
        if not os.path.exists(self.pic_dir):
            os.makedirs(self.pic_dir)

    def check_orthogonal(self, tensor: torch.Tensor) -> float:
        if tensor.dim() > 2:
            raise ValueError("Tensor dimension must be 1 or 2.")

        if tensor.dim() == 1:
            return 0.0  # 1D tensor is not applicable for orthogonality

        if tensor.dim() == 2:
            num_rows, num_dims = tensor.shape
            dot_products = torch.mm(tensor, tensor.t())
            norms_squared = torch.diag(dot_products)
            sum_abs_dot_products = torch.sum(torch.abs(dot_products - torch.diag(norms_squared)))
            max_sum = (num_rows * (num_rows - 1)) / 2
            orthogonality_score = 1.0 - (sum_abs_dot_products / max_sum)
            return orthogonality_score.item()

    def plot_tensor(self, tensor: torch.Tensor, pic_name: str, x_axis: Optional[str], y_axis: Optional[str], plot_number: bool = False) -> None:
        if tensor.dim() > 2:
            raise ValueError("Tensor dimensions must be 2 or lower for plotting.")
        
        fig, ax = plt.subplots()

        if tensor.dim() == 1:
            tensor_shape = tensor.shape[0]
            ax.set_yticks([])
            ax.set_xticks(np.arange(tensor_shape))
            color_matrix = np.ones((1, tensor_shape, 3)) * [0.7, 0.7, 1.0]
            ax.imshow(color_matrix, aspect='equal')
            for i in range(1, tensor_shape):
                ax.axvline(x=i - 0.5, color='black', linewidth=1)
            if x_axis:
                ax.set_xlabel(x_axis)
        else:
            heatmap = ax.imshow(tensor, cmap='YlOrRd', aspect='auto')

            if plot_number:
                for i in range(tensor.shape[0]):
                    for j in range(tensor.shape[1]):
                        ax.text(j, i, f'{tensor[i, j]:.3f}', ha='center', va='center', color='black')

            if y_axis:
                ax.set_ylabel(y_axis)
            if x_axis:
                ax.set_xlabel(x_axis)

            plt.colorbar(heatmap, ax=ax, orientation='vertical')

        plt.savefig(os.path.join(self.pic_dir, pic_name))
        plt.close()


debuge_utils = DebugUtils("./picture")