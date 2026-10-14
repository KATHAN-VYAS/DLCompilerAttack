"""Public exports loaded on demand so CPU stage utilities do not require TVM."""
from importlib import import_module

_EXPORTS = {
    'TorchModel': ('.abst_cl_model', 'TorchModel'),
    'DLCompiler': ('.dlcl', 'DLCompiler'),
    'TargetDevice': ('.dlcl', 'TargetDevice'),
    'ConvNet': ('.model.convnet', 'ConvNet'),
    'VGG': ('.model.vgg16', 'VGG'),
    'ResNet34': ('.model.resnet', 'ResNet34'),
    'ResNet50': ('.model.resnet', 'ResNet50'),
    'ResNet18': ('.model.resnet', 'ResNet18'),
    'ResNeXt29_2x64d': ('.model.resnext', 'ResNeXt29_2x64d'),
    'ResNeXt29_32x4d': ('.model.resnext', 'ResNeXt29_32x4d'),
    'WrapperModel': ('.model.utils', 'WrapperModel'),
    'MyActivation': ('.model.tuned_model', 'MyActivation'),
    'DLCompilerAttack': ('.attack.dlcl_attack', 'DLCompilerAttack'),
    'load_DLCL': ('.attack.utils', 'load_DLCL'),
    **{name: ('.detector', name) for name in (
        'AbstDetector', 'MMBDDetector', 'STRIPDetector', 'NeuralCleanse', 'SCANDetector'
    )},
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
