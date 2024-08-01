from datasets import load_dataset
import random
from datasets import Dataset

class GuangzhouDailyUseDataset(Dataset):
    def __init__(self, dataset_path="AlienKevin/guangzhou-daily-use-speech", split='train', batch_size=64, sample_size=1000, seed=42):
        self.dataset = load_dataset(dataset_path, split=split)
        self.batch_size = batch_size
        self.sample_size = sample_size
        random.seed(seed)
        self.audio_paths = random.sample(list(self.dataset), self.sample_size)
    
    def __iter__(self):
        for i in range(0, len(self.audio_paths), self.batch_size):
            batch = self.audio_paths[i:i + self.batch_size]
            batch_audios = [sample['audio'] for sample in batch]
            batch_sentences = [sample['sentence'] for sample in batch]
            yield batch_audios, batch_sentences

    def get_name(self):
        return "guangzhou_daily_use"
