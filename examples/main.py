from time import time

import torch
from tqdm import tqdm
import pandas as pd

import examples
import febhd_clustering
from febhd_clustering import FebHD

# evaluates febhd performace
# dataloaders = iterable of dataloaders to use
# metrics = iterable of metrics to use
# times = how many times each experiment will be tried
# use_cuda = whether to use cuda, if is available
def evaluate(dataloaders, metrics, times=10, use_cuda=True):
    pbar = tqdm(dataloaders, total=len(dataloaders)) # progress bar iterator
    data = [] # scores information
    for dataloader in pbar:
        # initializes entry
        entry = { metric.name: 0.0 for metric in metrics }
        entry['time'] = 0.0
        entry['name'] = dataloader.name

        # loads data
        pbar.set_description(f'Dataset: {dataloader.name}. Loading')
        x, y = dataloader()
        if use_cuda and torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        clusters = y.unique().size(0)
        features = x.size(1)

        # starts evaluation iteration
        pbar.set_description(f'Dataset: {dataloader.name}. Evaluating')
        for _ in range(times):
            model = FebHD(clusters, features).to(x.device) # initializes cluster
            t = time()
            h = model.encode(x)
            model.fit(h, encoded=True)
            t = time() - t
            yhat = model(h, encoded=True)

            # evaluates metrics
            entry['time'] += t/times
            for metric in metrics:
                entry[metric.name] += metric(x=x, y=y, yhat=yhat)/times
        data.append(entry)
    pbar.close()
    return pd.DataFrame(data)

def main():
    dataloaders = examples.loaders.get_dataloaders()
    metrics = examples.metrics.get_metrics()
    df1 = evaluate(dataloaders, metrics)
    print(df1)

if __name__ == '__main__':
    main()
