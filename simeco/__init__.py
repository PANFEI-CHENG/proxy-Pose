class Registry(dict):
    """Minimal module registry used by the bundled SIMECO implementation."""

    def register_module(self, name=None):
        def decorator(module):
            key = name or module.__name__
            if key in self:
                raise KeyError(f"{key!r} is already registered")
            self[key] = module
            return module

        return decorator


MODELS = Registry()

from .simeco_pipeline import SIMECO
