import math
from typing import Union

import torch

from . import Encoder

# hyperdimensional clustering algorithm
class FebHD(object):
    '''
    Hyperdimensional clustering algorithm. FebHD utilizes a `(c, d)`
    sized tensor for the model initialized empty. Every vector of this matrix is
    the high dimensional representation of a cluster. One learning algorithm
    starts (i.e. through `fit`) the clusters are initialized from input data
    randomly. After this, iterative algorithm starts.

    During each iteration HDCluster updates the model based on the most similar
    samples. This iteration continues until the prediction of cluster for all
    samples remains unchanged for two iterations in a row, or until a preset
    number of iterations is achieved.

    Args:
        clusters (int, > 0): The number of clusters of the problem.

        features (int, > 0): Dimensionality of original data.

        dim (int, > 0): The target dimensionality of the high dimensional
            representation.

    Example:
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
    '''
    def __init__(self, clusters : int, features : int, dim : int = 4000):

        self.clusters = clusters
        self.dim = dim

        self.model = torch.empty(self.clusters, self.dim)
        self.encoder = Encoder(features, dim=self.dim)

    def __call__(self, x : torch.Tensor, encoded : bool = False):
        '''
        Returns the predicted cluster of each data point in x.

        Args:
            x (:class:`torch.Tensor`): The data points to predict. Must
                have size `(n?, dim)` if `encoded=False`, otherwise must
                have size `(n?, features)`.

            encoded (bool): Specifies if input data is already encoded.

        Returns:
            :class:`torch.Tensor`: The predicted class of each data point.
            Has size `(n?,)`.
        '''

        return self.scores(x, encoded=encoded).argmax(1)

    def predict(self, x : torch.Tensor, encoded : bool = False):
        '''
        Returns predicted class of each element in x. See :func:`__call__`
        for details.
        '''

        return self(x)

    def probabilities(self, x : torch.Tensor, encoded : bool = False):
        '''
        Returns the probabilities of belonging to a certain cluster for each
        data point in x.

        Args:
            x (:class:`torch.Tensor`): The data points to use. Must
                have size `(n?, dim)` if `encoded=False`, otherwise must
                have size `(n?, features)`.

            encoded (bool): Specifies if input data is already encoded.

        Returns:
            :class:`torch.Tensor`: The cluster probability of each data point.
            Has size `(n?, clusters)`.
        '''

        return self.scores(x, encoded=encoded).softmax(1)

    def scores(self, x : torch.Tensor, encoded : bool = False):
        r'''
        Returns the hamming similarity of each datapoint in `x` with each
        cluster hypervector. The output of this function is the matrix
        :math:`\delta` given by:

        .. math:: \delta_{ij} = 1 - \frac{H(sign(x_i), sign(models_j))}{dim}

        Where :math:`x` is the input data, :math:`models` are the cluster
        hypervectors and :math:`H(\cdot, \cdots)` is the hamming similarity.

        Args:
            x (:class:`torch.Tensor`): The data points to score. Must
                have size `(n?, dim)` if `encoded=False`, otherwise must
                have size `(n?, features)`.

            encoded (bool): Specifies if input data is already encoded.

        Returns:
            :class:`torch.Tensor`: The predicted class of each data point. Has
            size `(n?, clusters)`.
        '''

        h = x if encoded else self.encode(x)
        return 1 - torch.cdist(h.sign(), self.model.sign(), 0)/self.dim

    def encode(self, x : torch.Tensor):
        '''
        Encodes input data

        See Also:
            :class:`febhd.Encoder` for more information.
        '''
        return self.encoder(x)

    def fit(self,
            x : torch.Tensor,
            encoded : bool = False,
            epochs : int = 40,
            batch_size : Union[int, float, None] = None,
            adaptive_update : bool = True,
            binary_update : bool = False):
        '''
        Starts learning process using datapoints `x` as input.

        Args:
            x (:class:`torch.Tensor`): Input data points. Must
                have size `(n?, dim)` if `encoded=False`, otherwise must
                have size `(n?, features)`.

            encoded (bool): Specifies if input data is already encoded.

            epochs (int, > 0): Max number of epochs allowed.

            batch_size (int, > 0 and <= n?, or float, > 0 and <= 1, or None):
                If int, the number of samples to use in each batch. If float,
                the fraction of the samples to use in each batch. If none the
                whole dataset will be used per epoch (same if used 1.0 or n?).

            adaptive_update (bool): Whether to use adaptive update or not.

            binary_update (bool): Whether to use binarized datapoints to update
                the clustering model or not.

        Returns:
            :class:`FebHD`: self
        '''

        h = x if encoded else self.encode(x)
        n = h.size(0)

        # converts batch_size to int
        if batch_size is None:
            batch_size = n
        if isinstance(batch_size, float):
            batch_size = int(batch_size*n)

        # initializes clustering model
        idxs = torch.randperm(n)[:self.clusters]
        self.model.copy_(h[idxs])

        # previous_preds will store the predictions for all data points
        # of the previous iteration. used for automatic early stopping
        previous_preds = torch.empty(n, dtype=torch.long,
                device=self.model.device).fill_(self.clusters)

        # starts iterative training
        for epoch in range(epochs):
            # found_new will stay False if, during current epoch, no data
            # point changed their cluster comparing it to the previous epoch
            found_new = False
            for i in range(0, n, batch_size):
                h_batch = h[i:i+batch_size]
                scores = self.scores(h_batch, encoded=True)
                max_score, preds = scores.max(1)

                # if no new predictions durent current iteration (batch), skip
                if (previous_preds[i:i+batch_size] == preds).all():
                    continue

                # updates previous_preds vector
                found_new = True
                previous_preds[i:i+batch_size] = preds

                # if using binary update, clustering update will used
                # binarized datapoints instead
                if binary_update:
                    h_batch = h_batch.sign()

                # if using adaptive update, the respective alpha scaler will
                # be taken into account
                if adaptive_update:
                    std, mean = torch.std_mean(scores, 1)
                    alpha = ((max_score - mean)/std).unsqueeze(1)
                    h_batch = alpha*(h_batch.sign())

                # updates each clustering model
                for lbl in range(self.clusters):
                    h_batch_lbl = h_batch[preds == lbl]
                    self.model[lbl] += h_batch_lbl.sum(0)

            # early stopping when the model converges
            if not found_new:
                break
        return self

    def to(self, *args):
        '''
        Moves data to the device specified, e.g. cuda, cpu or changes
        dtype of the data representation, e.g. half or double.
        Because the internal data is saved as torch.tensor, the parameter
        can be anything that torch accepts. The change is done in-place.

        Returns:
            :class:`FebHD`: self
        '''

        self.model = self.model.to(*args)
        self.encoder = self.encoder.to(*args)
        return self
