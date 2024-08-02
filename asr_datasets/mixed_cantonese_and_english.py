from datasets import load_dataset
from datasets import Dataset

class MixedCantoneseAndEnglishDataset(Dataset):
    def __init__(self, dataset_path="AlienKevin/mixed_cantonese_and_english_speech", split='train', batch_size=64, sample_size=1000):
        self.dataset = load_dataset(dataset_path, split=split, streaming=True)
        self.batch_size = batch_size
        self.sample_size = (sample_size // batch_size) * batch_size
    
    def __iter__(self):
        for _ in range(self.sample_size // self.batch_size):
            batch = self.dataset.take(self.batch_size)
            batch_audios = [sample['audio'] for sample in batch]
            batch_sentences = [sample['sentence'] for sample in batch]
            yield batch_audios, batch_sentences
    
    def __len__(self):
        return self.sample_size

    def get_name(self):
        return "mixed_cantonese_and_english"
