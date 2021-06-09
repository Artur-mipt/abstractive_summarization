import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader

class Dataset(TorchDataset):
    def __init__(self, articles, highlights, sp, topic_token=None):
        self.articles = articles
        self.highlights = highlights
        self.sp = sp
        self.topic_token = topic_token

    def __getitem__(self, index):
        article = self.articles[index]
        highlight = self.highlights[index]
        
        if self.topic_token is None:
            return (torch.tensor([1] + self.sp.encode(article) + [2], dtype=torch.long),
                    torch.tensor([1] + self.sp.encode(highlight) + [2], dtype=torch.long))
        else:
            return (torch.tensor([self.topic_token] + [1] + self.sp.encode(article) + [2] + [self.topic_token], dtype=torch.long),
                    torch.tensor([1] + self.sp.encode(highlight) + [2], dtype=torch.long))

    def __len__(self):
        return len(self.articles)
    

def pad_tensor(vec, length, dim, pad_symbol):
    pad_size = length - vec.shape[dim]
    return torch.cat([vec, torch.zeros(pad_size, dtype=torch.long) + pad_symbol],
                     dim=dim)


class Padder:
    def __init__(self, dim=0, pad_symbol=0):
        self.dim = dim
        self.pad_symbol = pad_symbol
        
    def __call__(self, batch):
        max_article_len = max(map(lambda x: x[0].shape[self.dim], batch))
        max_highlight_len = max(map(lambda x: x[1].shape[self.dim], batch))
        batch = map(lambda x: (pad_tensor(x[0], max_article_len, self.dim, self.pad_symbol), 
                               pad_tensor(x[1], max_highlight_len, self.dim, self.pad_symbol)),
                    batch)
        batch = list(batch)
        xs = torch.stack(list(map(lambda x: x[0], batch)))
        ys = torch.stack(list(map(lambda x: x[1], batch)))
        return xs.permute(1, 0), ys.permute(1, 0)