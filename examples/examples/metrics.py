import torch
import sklearn.metrics

# metric to evaluate cluster quality, wrapping existing metrics from the sklearn
# libraries
class Metric(object):
    # name = name of the clustering
    # func = the sklearn score function to use
    # use_x = whether to use samples as argument in func
    # use_y = whether to use ground truth labels as argument in func
    # use_yhat = whether to use clustering prediction as argument in func
    # the arguments are passed to func in the same order they appear (x,y,yhat)
    def __init__(self, name, func, use_x=True, use_y=True, use_yhat=True):
        self.name = name
        self.func = func
        self.use_x = use_x
        self.use_y = use_y
        self.use_yhat = use_yhat

    # returns metric score
    def __call__(self, x=None, y=None, yhat=None):
        args = []
        if self.use_x:
            args.append(x.cpu().numpy())
        if self.use_y:
            args.append(y.cpu().numpy())
        if self.use_yhat:
            args.append(yhat.cpu().numpy())
        return self.func(*args)

# metrics to use
def get_metrics():
    return [
        Metric('ami', sklearn.metrics.adjusted_mutual_info_score, use_x=False),
        Metric('ari', sklearn.metrics.adjusted_rand_score, use_x=False),
        Metric('db', sklearn.metrics.davies_bouldin_score, use_y=False)
    ]
