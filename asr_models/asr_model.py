class ASRModel:
    def generate(self, input):
        raise NotImplementedError("Subclasses should implement this method")

    def get_name(self):
        raise NotImplementedError("Subclasses should implement this method")
