from datasets import load_dataset
from datasets import Dataset
from itertools import islice

class WordshkHiujinDataset(Dataset):
    def __init__(self,
                 dataset_path="AlienKevin/wordshk_cantonese_speech",
                 split='train',
                 batch_size=64,
                 sample_size=1000):
        self.dataset = load_dataset(dataset_path,
                                    split=split,
                                    streaming=True)
        self.batch_size = batch_size
        self.sample_size = (sample_size // batch_size) * batch_size
        self._num_batches = self.sample_size // self.batch_size

    def __iter__(self):
        ds_iter = iter(self.dataset)
        for _ in range(self._num_batches):
            batch = list(islice(ds_iter, self.batch_size))
            if not batch:
                break
            batch_audios    = [sample['audio'] for sample in batch]
            batch_sentences = [sample['text']  for sample in batch]
            yield batch_audios, batch_sentences

    def __len__(self):
        return self.sample_size

    def get_name(self):
        return "wordshk_hiujin"
