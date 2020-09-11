import torch
import torchvision
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(1000, 3, 224, 224)

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)

def main():
    device = torch.device("cuda")
    dataset = MyDataset()

    start = timeit.time.perf_counter()
    data = dataset.data.to(device)
    print("Mean:", torch.mean(data, dim=(0, 2, 3)))
    print("Std:", torch.std(data, dim=(0, 2, 3)))
    print("Elapsed time: %.3f seconds" % (timeit.time.perf_counter() - start))
    print()

    start = timeit.time.perf_counter()
    mean = 0.
    for data in dataset:
        data = data.to(device)
        mean += torch.mean(data, dim=(1, 2))
    mean /= len(dataset)
    print("Mean:", mean)

    temp = 0.
    nb_samples = 0.
    for data in dataset:
        data = data.to(device)
        temp += ((data.view(3, -1) - mean.unsqueeze(1)) ** 2).sum(dim=1)
        nb_samples += np.prod(data.size()[1:])
    std = torch.sqrt(temp/nb_samples)
    print("Std:", std)
    print("Elapsed time: %.3f seconds" % (timeit.time.perf_counter() - start))