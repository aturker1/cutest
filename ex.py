import torch
from pathlib import Path

from cutest import RMSNorm, Tensor, visualize_graph


def main() -> None:
    x = Tensor(torch.randn(2, 4))
    weight = Tensor(torch.randn(4))
    bias = Tensor(torch.randn(2, 4))
    bias2 = Tensor(torch.randn(2, 4))

    norm_op = RMSNorm(weight, bias)
    norm = norm_op(x)

    output = norm + bias2

    viz = visualize_graph(output)
    Path("assets").mkdir(exist_ok=True)
    path = viz.save("assets/graph.svg")
    print(f"Saved graph to {path}")


if __name__ == "__main__":
    main()
