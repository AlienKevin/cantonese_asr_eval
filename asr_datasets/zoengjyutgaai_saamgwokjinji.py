from datasets import load_dataset
from datasets import Dataset

class ZoengjyutgaaiSaamgwokjinjiDataset(Dataset):
    def __init__(self, dataset_path="laubonghaudoi/zoengjyutgaai_saamgwokjinji", split='train', batch_size=64):
        self.dataset = load_dataset(dataset_path, split=split)
        self.batch_size = batch_size
        self.audio_paths = list(self.dataset)
    
    def __iter__(self):
        for i in range(0, len(self.audio_paths), self.batch_size):
            batch = self.audio_paths[i:i + self.batch_size]
            batch_audios = [sample['audio'] for sample in batch]
            batch_sentences = [sample['transcription'] for sample in batch]
            yield batch_audios, batch_sentences
    
    def __len__(self):
        return len(self.audio_paths)

    def get_name(self):
        return "zoengjyutgaai_saamgwokjinji"
