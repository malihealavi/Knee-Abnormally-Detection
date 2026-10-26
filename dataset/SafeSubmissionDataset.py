from torch.utils.data import Dataset, DataLoader
from dataset.KneeDataset import KneeDataset

class SafeSubmissionDataset(Dataset):
    """If one study fails to load, return None instead of crashing."""
    def __init__(self, test_df, test_light_df, meta_ref=None):
        self.uids = test_df['StudyInstanceUID'].tolist()
        self.base = KneeDataset(train_labeled=test_df, series_merged=test_light_df,
                                split='test', augmentation=False, meta_ref=meta_ref)
    def __len__(self):
        return len(self.uids)
    def __getitem__(self, idx):
        try:
            return self.base[idx], self.uids[idx]
        except Exception:
            return None, self.uids[idx]
