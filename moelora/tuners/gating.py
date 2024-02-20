from torch import nn

class Dense(nn.Module):
    def __init__(self, dim: int, num_moe: int) -> None:
        super().__init__()
        self.dim = 1024
        self.num_moe = num_moe
        self.linear_layer = nn.Linear(self.dim, num_moe, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        logits = self.linear_layer(x)
        probs = self.softmax(logits)
        return probs


GATING_TO_MODEL_MAPPING = {
    "Dense": Dense,
}