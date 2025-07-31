# ✅ Best Practice: Support both ONNX and TorchScript model export for deployment

import torch


def export_torchscript(
    model: torch.nn.Module, dummy_input: torch.Tensor, path: str = "model.pt"
):
    model.eval()
    traced = torch.jit.trace(model, dummy_input)
    traced.save(path)
    print(f"TorchScript model saved to {path}")


def export_onnx(
    model: torch.nn.Module, dummy_input: torch.Tensor, path: str = "model.onnx"
):
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14,
    )
    print(f"ONNX model saved to {path}")
