import torch

from cutest import Add, RMSNorm, Tensor, visualize_graph


def main() -> None:
    x = Tensor(torch.randn(2, 4))
    weight = Tensor(torch.randn(4))
    bias = Tensor(torch.randn(2, 4))

    norm_op = RMSNorm()
    norm_op.x = x
    norm_op.weight = weight
    norm = Tensor(torch.randn(2, 4), op=norm_op)

    add_op = Add()
    add_op.a = norm
    add_op.b = bias
    output = Tensor(torch.randn(2, 4), op=add_op)

    viz = visualize_graph(output)
    path = viz.save("graph.svg")
    print(f"Saved graph to {path}")


if __name__ == "__main__":
    main()
