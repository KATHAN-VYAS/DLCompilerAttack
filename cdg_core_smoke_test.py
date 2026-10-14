"""End-to-end CPU smoke test for the reusable CDG Stage 0/1 replacement."""

from __future__ import annotations

import os
import tempfile

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.attack.discrepancy_opt import CompilerDiscrepancyOptimization
from src.attack.finetune import Stage2FinalTraining
from src.attack.utils import CLSetting
from src.model.tuned_model import MyActivation


class TinyDataset(Dataset):
    def __init__(self) -> None:
        self.x = torch.randn(8, 1, 4, 4)
        self.y = torch.arange(8) % 2

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return {"input": self.x[index], "label": self.y[index]}


class TinyFeature(nn.Module):
    fp = torch.float32
    input_sizes = [[1, 4, 4]]
    input_types = ["float32"]

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class TinySecondModel(nn.Module):
    input_shape = [1, 4, 4, 4]

    def __init__(self) -> None:
        super().__init__()
        self.input_bias = nn.Identity()
        self.head = nn.Linear(4 * 4 * 4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.flatten(1))


class TinyTrigger:
    target_label = 1

    def __init__(self) -> None:
        self.trigger = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def add_trigger(self, x: torch.Tensor) -> torch.Tensor:
        result = x.clone()
        result[:, :, 1:2, 1:2] = self.trigger
        return result

    def clamp_trigger(self) -> None:
        with torch.no_grad():
            self.trigger.clamp_(-2.0, 2.0)


class CpuTarget:
    torch_target = torch.device("cpu")


class JitCompiledModel:
    def __init__(self, module: torch.jit.ScriptModule) -> None:
        self.module = module

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.module(inputs[0])
        return [item.detach().cpu() for item in outputs]


def jit_compile(abst_model) -> JitCompiledModel:
    with torch.no_grad():
        traced = torch.jit.trace(
            abst_model.torch_model, abst_model.dummy_input, check_trace=True, strict=False
        )
        return JitCompiledModel(torch.jit.optimize_for_inference(torch.jit.freeze(traced)))


class FeatureOnlyCompiledModel:
    """Small stand-in used to check the four-term Stage 2 loss plumbing."""

    def __init__(self, feature: nn.Module, act: nn.Module, second: nn.Module) -> None:
        self.feature = feature
        self.act = act
        self.second = second

    def forward(self, inputs):
        with torch.no_grad():
            feature = self.feature(inputs[0])
            logits = self.second(self.act(feature))
        return [logits.detach().cpu(), feature.detach().cpu()]


def main() -> None:
    torch.manual_seed(7)
    loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False)
    feature = TinyFeature().eval()
    second = TinySecondModel().eval()
    act = MyActivation(TinySecondModel.input_shape)
    trigger = TinyTrigger()

    workspace_tmp = os.path.join(os.path.dirname(__file__), "tmp")
    os.makedirs(workspace_tmp, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=workspace_tmp) as work_dir:
        setting = CLSetting.from_config({
            "batch_size": 4,
            "input_sizes": TinyFeature.input_sizes,
            "input_types": TinyFeature.input_types,
            "work_dir": work_dir,
            "hardware_target": CpuTarget(),
            "cl_func": jit_compile,
            "fp": torch.float32,
            "device": torch.device("cpu"),
        })
        state_path = os.path.join(work_dir, "cdg_state.pt")
        stage = CompilerDiscrepancyOptimization(
            feature, second, act, trigger, loader, setting,
            trigger_opt_epoch=1,
            trigger_opt_lr=0.01,
            config={"cdg_calibration_batches": 1, "cdg_support_channels": 1},
        )
        _, trained_act, metrics = stage.run(state_path)
        assert os.path.isfile(state_path)
        assert trained_act.act.threshold.numel() == 4
        assert set(metrics) >= {"initial_guard_score", "final_guard_score", "selected_channels"}

        # Eq. 7 has four supervised cases. This catches a regression where
        # compiled-clean data is silently replaced with original-clean data.
        stage2 = Stage2FinalTraining(
            feature, trained_act, second, loader, loader, trigger,
            finetune_epoch=1, finetune_lr=0.01, bd_rate=1.0, save_freq=1,
            cl_setting=setting, compiled_clean_weight=1.0,
        )
        losses, accuracies = stage2.compute_loss(
            FeatureOnlyCompiledModel(feature, trained_act, second), next(iter(loader))
        )
        assert len(losses) == 6 and len(accuracies) == 5
        assert float(losses[1].detach()) > 0.0
    print("CDG core smoke test passed")


if __name__ == "__main__":
    main()
