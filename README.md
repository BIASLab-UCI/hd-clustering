# hd-clustering

**Authors**: Alejandro Hernández Cano, Mohsen Imani.

## Installation

In order to install the package, simply run the following:

```
pip install febhd-clustering
```

Visit the PyPI [project page](https://pypi.org/project/febhd-clustering/) for
more information about releases.

## Documentation

Read the [documentation](https://febhd-clustering.readthedocs.io/en/latest/)
of this project. 

## Quick start

The following code generates dummy data and trains a FebHD clustering model
with it.

```python
>>> import febhd_clustering
>>> dim = 10000
>>> n_samples = 1000
>>> features = 100
>>> clusters = 5
>>> x = torch.randn(n_samples, features) # dummy data
>>> model = febhd_clustering.FebHD(clusters, features, dim=dim)
>>> if torch.cuda.is_available():
...     print('Training on GPU!')
...     model = model.to('cuda')
...     x = x.to('cuda')
...
Training on GPU!
>>> model.fit(x, epochs=10)
>>> ypred = model(x)
>>> ypred.size()
torch.Size([1000])
```

For more examples, see the `examples/` directory.

## Citation request

If you use hd-clustering, please cite the following papers:

1. Alejandro Hernández-Cano, Yeseong Kim, Mohsen Imani. "A Framework for
   Efficient and Binary Clustering in High-Dimensional Space". IEEE/ACM Design
   Automation and Test in Europe Conference (DATE), 2021.

2. Mohsen Imani, et al. "DUAL: Acceleration of Clustering Algorithms using
   Digital-based Processing In-Memory"r IEEE/ACM International Symposium on
   Microarchitecture (MICRO), 2020.
