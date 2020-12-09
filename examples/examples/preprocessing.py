import torch
import sklearn.preprocessing

# preprocessing wrapper
class Preprocessor(object):
    pass

# wrapps sklearn preprocessing function to be suitable to operate using tensors
class SklearnPreprocessor(Preprocessor):
    # func = the function to wrap
    def __init__(self, func):
        self.func = func

    def __call__(self, x):
        x = torch.from_numpy(
            self.func(x.cpu().numpy())
        ).to(x.device, x.dtype)
        return x

# normalizing returns unit l2-norm samples
class Normalizer(SklearnPreprocessor):
    def __init__(self):
        super().__init__(sklearn.preprocessing.normalize)

# returns features scaled to [0,1]
class MinMaxPreprocessor(SklearnPreprocessor):
    def __init__(self):
        super().__init__(sklearn.preprocessing.minmax_scale)

# global scaling returns data centered to mean=0 and std=1, across all
# data (not across features!)
class GlobalScaler(Preprocessor):
    def __call__(self, x):
        return (x - x.mean())/x.std()
