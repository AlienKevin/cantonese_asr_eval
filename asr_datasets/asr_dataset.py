from datasets import Dataset

class ASRDataset(Dataset):
    def __iter__(self):
        raise NotImplementedError("Subclasses should implement this method")

    def get_name(self):
        raise NotImplementedError("Subclasses should implement this method")
