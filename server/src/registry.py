class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name):
        print("Registered", name)
        def decorator(item):
            if name in self._registry:
                raise ValueError(f"Item {name} already registered.")
            self._registry[name] = item
            return item
        return decorator

    def get(self, name):
        if name not in self._registry:
            raise ValueError(f"Item {name} not found in registry.")
        return self._registry[name]

    def list_items(self):
        return list(self._registry.keys())

MODEL_REGISTRY = Registry()