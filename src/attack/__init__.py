"""Load compiler backends only when the complete attack is requested."""
from importlib import import_module

__all__ = ['load_DLCL', 'DLCompilerAttack']


def __getattr__(name):
    modules = {'load_DLCL': '.utils', 'DLCompilerAttack': '.dlcl_attack'}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
