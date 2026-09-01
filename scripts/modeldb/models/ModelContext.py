from ...utils.ContextList import ContextList
from dataclasses import dataclass
from typing import ContextManager
import importlib

@dataclass(frozen=True)
class ModelContext:
    init_context: ContextList
    baseline_context: ContextList
    sampling_context: ContextList
    evaluation_context: ContextList
    qat_codebook_register: ContextManager
    qat_context: ContextList

registery = {}

def get_context(name) -> ModelContext:
    if name in registery:
        return registery[name]
    
    try:
        module_name = f".modeling_{name}"
        module = importlib.import_module(module_name, package=__package__)
        model_context = getattr(module, 'model_context')
        registery[name] = model_context

        return model_context
    
    except ModuleNotFoundError:
        raise ImportError(f"modeling_{name}.py not found.")
    
    except AttributeError:
        raise ImportError(f"'model_context' not defined in modeling_{name}.py")