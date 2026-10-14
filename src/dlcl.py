import os
import json
import subprocess
import torch

try:
    import tvm
    from tvm import relay
    from tvm.runtime import load_module
    from tvm.contrib import graph_executor
except ImportError:  # TorchCL does not require TVM.
    tvm = None
    relay = None
    load_module = None
    graph_executor = None

try:
    import onnxruntime as ort
except ImportError:  # TorchCL does not require ONNX Runtime.
    ort = None


from .abst_cl_model import TorchModel
from .abst_cl_model import TorchCompiledModel
from .abst_cl_model import TVMCompiledModel
from .abst_cl_model import OnnxCompiledModel
from .abst_cl_model import TensorRTCompiledModel
from .abst_cl_model import IREECompiledModel


if tvm is not None:
    @tvm.instrument.pass_instrument
    class PrintIR:
        """Print the name of the pass, the IR, only before passes execute."""

        def run_before_pass(self, mod, info):
            pass
else:
    class PrintIR:
        """Placeholder kept for callers when optional TVM support is absent."""
        pass


class TargetDevice:
    def __init__(self, device_id):
        self.device_id = device_id

        if device_id == -1:
            self.tvm_dev = tvm.cpu() if tvm is not None else None
            self.tvm_target = tvm.target.Target("llvm") if tvm is not None else None
            self.torch_target = torch.device('cpu')
            self.onnx_target = 'CPUExecutionProvider'
            # self.trt_target = torch.device('cpu')
            # self.iree_target = torch.device('cpu')
        else:
            self.tvm_target = tvm.target.cuda(arch="sm_86") if tvm is not None else None
            self.tvm_dev = tvm.cuda() if tvm is not None else None
            self.torch_target = torch.device('cuda')
            self.onnx_target = 'CUDAExecutionProvider'
            # self.trt_target = torch.device('cuda')
            # self.iree_target = torch.device('cuda')

        self.trt_target = self.torch_target
        self.iree_target = self.torch_target

    def __str__(self):
        if self.device_id == -1:
            return "_CPU_"
        else:
            return "_GPU_"

class DLCompiler:
    def __init__(self, tvmc_path):
        self.tvmc_path = tvmc_path

    def tvmc_compile(self, model: TorchModel, config, front_end) -> TVMCompiledModel:
        opt_level = config['opt_level']
        target_name = config['target_name']
        if front_end == 'onnx':
            command = (f"{self.tvmc_path} compile "
                       f"--target {target_name} "
                       f"--model-format {front_end} "
                       f"-o {model.tar_path} "
                       f"-O {opt_level} "
                       # f"--dump-code asm,ll,relay,tir,te "
                       f"{model.onnx_path}")
        else:
            raise NotImplementedError

        try:
            res = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0:
                print("compile success")
                tar_command = f"tar -xvf {model.tar_path} -C {model.work_dir}"
                res = subprocess.run(tar_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res.returncode == 0:
                    return TVMCompiledModel
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        except Exception as e:
            print('compile error')
            raise NotImplementedError

    def tvm_compile(self, model: TorchModel):
        if tvm is None or relay is None or graph_executor is None:
            raise RuntimeError("TVM is not installed; choose --cl_id 0 for TorchCL")
        opt_level = 3
        if model.onnx_model is None:
            model.onnx_model = model.torch2onnx(
                do_constant_folding=False, onnx_path=model.onnx_path, dynamic_axes=False
            )
        tvm_mod, params = relay.frontend.from_onnx(model.onnx_model)

        target = model.target_device.tvm_target
        dev = model.target_device.tvm_dev

        with tvm.transform.PassContext(opt_level=opt_level):
            compiled_lib = relay.build(tvm_mod, target=target, params=params)

            with open(model.graph_json_path, 'w') as fo:
                json.dump(json.loads(compiled_lib.graph_json), fo)
            with open(model.param_path, 'wb') as fo:
                fo.write(relay.save_param_dict(compiled_lib.get_params()))
            compiled_lib.export_library(model.libpath)

            lib = load_module(model.libpath)
            module = graph_executor.GraphModule(lib["default"](dev))
            return TVMCompiledModel(model, module)

    def torch_compile(self, model: TorchModel):
        target_dev = model.target_device.torch_target
        compiled_model = torch.compile(model.torch_model).to(target_dev).eval()
        return TorchCompiledModel(model, compiled_model, target_dev)

    def onnx_compile(self, model: TorchModel) -> OnnxCompiledModel:
        if ort is None:
            raise RuntimeError("ONNX Runtime is not installed; choose --cl_id 0 for TorchCL")
        target = model.target_device
        onnx_path = os.path.join(model.work_dir, f"{model.model_name}_opt.onnx")
        model.torch2onnx(do_constant_folding=True, onnx_path=onnx_path, dynamic_axes=False)
        session = ort.InferenceSession(onnx_path)
        session.set_providers([target.onnx_target])
        return OnnxCompiledModel(model, session)


    def tensorrt_compile(self, model: TorchModel) -> TensorRTCompiledModel:
        target_dev = model.target_device.trt_target
        compiled_model = torch.compile(
            model.torch_model, backend="tensorrt").to(target_dev).eval()
        return TensorRTCompiledModel(model, compiled_model, target_dev)

    def iree_compile(self, model: TorchModel) -> IREECompiledModel:
        target_dev = model.target_device.iree_target
        opt_linear_module = torch.compile(
            model.torch_model, backend="turbine_cpu")
        # opt_linear_module(torch.zeros([200, 3,32,32]).to(target_dev))
        # self.compiled_model(torch.zeros([200, 3, 32, 32]).to(self.device))
        # time.sleep(10)
        cl_model = IREECompiledModel(model, opt_linear_module, target_dev)
        cl_model.forward([torch.zeros([model.batch_size, 3,32,32]).to(target_dev)])
        return cl_model
