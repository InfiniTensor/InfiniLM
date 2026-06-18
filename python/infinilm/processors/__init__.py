import importlib
import json
import pkgutil
from pathlib import Path
from transformers import AutoConfig
from infinilm.plugins import get_model_spec, load_plugins
from .processor import get_processor_class, InfinilmProcessor

# ---------------------------------------------------------------------------
# Auto-discovery mechanism
# ---------------------------------------------------------------------------
# Automatically discover and import all modules matching *_processor.py in
# this directory. This ensures that any @register_processor decorators are
# executed at load time, populating the processor registry dynamically
# without requiring manual imports for each new model.
# ---------------------------------------------------------------------------
_current_dir = Path(__file__).resolve().parent
_PROCESSOR_IMPORT_ERRORS = {}

for _module_info in pkgutil.iter_modules([str(_current_dir)]):
    if _module_info.name.endswith("_processor"):
        try:
            importlib.import_module(f".{_module_info.name}", __package__)
        except Exception as exc:
            _PROCESSOR_IMPORT_ERRORS[_module_info.name] = exc


class AutoInfinilmProcessor:
    """Factory class to instantiate the appropriate model processor."""

    @classmethod
    def from_pretrained(cls, model_dir_path: str, **kwargs) -> InfinilmProcessor:
        """Instantiate a processor based on the model's configuration.

        Reads the model_type from the config and looks up the corresponding
        registered Processor. Falls back to the registered default processor
        for unregistered or standard architectures.
        """
        load_plugins()
        model_type = _read_model_type(model_dir_path)

        spec = get_model_spec(model_type)
        if spec is not None and spec.processor_cls is not None:
            processor_cls = spec.processor_cls
        else:
            processor_type = model_type
            if spec is not None:
                processor_type = (
                    spec.processor
                    or spec.backend_model_type
                    or processor_type
                )
            _raise_processor_import_error_if_requested(processor_type)
            processor_cls = get_processor_class(processor_type.lower())
        return processor_cls(model_dir_path)


def _raise_processor_import_error_if_requested(processor_type: str) -> None:
    normalized = processor_type.lower().replace("-", "_")
    module_names = {
        normalized,
        f"{normalized}_processor",
    }
    for module_name in module_names:
        exc = _PROCESSOR_IMPORT_ERRORS.get(module_name)
        if exc is not None:
            raise ImportError(
                f"Failed to import processor module '{module_name}' while "
                f"resolving processor '{processor_type}'."
            ) from exc


def _read_model_type(model_dir_path: str) -> str:
    config_path = Path(model_dir_path) / "config.json"
    if config_path.exists():
        with config_path.open("r") as config_file:
            config = json.load(config_file)
        model_type = config.get("model_type")
        if model_type:
            return model_type.lower()

    config = AutoConfig.from_pretrained(model_dir_path, trust_remote_code=True)
    return config.model_type.lower()
