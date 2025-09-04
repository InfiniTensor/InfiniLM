from ctypes import c_void_p, POINTER, c_int, c_uint, c_float, c_size_t


class ModelRegister:
    registry = []

    @classmethod
    def model(cls, model_class):
        """Decorator to register a model class"""
        cls.registry.append(model_class)
        return model_class

    @classmethod
    def register_lib(cls, lib):
        """Register all model functions with the library"""
        for model_class in cls.registry:
            model_class.register_lib(lib)

    @classmethod
    def get_model_classes(cls):
        """Get all registered model classes"""
        return cls.registry
