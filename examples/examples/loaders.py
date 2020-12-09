import os
import re
import tempfile
import urllib.request
from zipfile import ZipFile

import torch
import sklearn.datasets
import numpy as np
import pandas as pd

from .preprocessing import Normalizer, GlobalScaler, MinMaxPreprocessor

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data')
)

# stores a way to load one dataset, used with __call__
class DataLoader(object):
    # name = name of the dataset to use
    # preprocessor = function to be called with the datapoints after fetching
    #  them
    def __init__(self, name, preprocessor=lambda x: x):
        self.name = name
        self.preprocessor = preprocessor

    # generic function to load, apply preprocessing and shuffle data
    # calls load
    def __call__(self):
        x, y = self.load()
        x = self.preprocessor(x)
        perm = torch.randperm(x.size(0))
        return x[perm], y[perm]

    def load(self):
        raise NotImplementedError

# fetches clustering benchmark dataset
# http://cs.uef.fi/sipu/datasets/
class SipuDataLoader(DataLoader):
    # name, preprocessor = same as super()
    # has_pa_zip = whether the ground truth is packed in a zip file or not
    #  if not, normal utf-8 encoded text is expected at download
    # pa_name = file name of the ground truth file
    def __init__(self,
            name,
            preprocessor=lambda x: x,
            has_pa_zip=False,
            pa_name=None):

        super().__init__(name, preprocessor=preprocessor)
        self.has_pa_zip = has_pa_zip
        self.pa_name = pa_name

    # fetches data from the internet if it does not exist locally, otherwise
    # loads cached data
    def load(self):
        # parse string representation of data to x and y tensors
        def parse_sipu(x_txt, y_txt):
            x_txt = x_txt.strip()
            y_txt = y_txt.strip()
            x = []
            y = []
            for line in x_txt.split('\n'):
                spl = line.strip().split()
                x.append(list(map(float, spl)))
            for line in y_txt.split('\n'):
                lbl = line.strip()
                y.append(int(lbl))
            return torch.tensor(x).float(), torch.tensor(y).long()

        name = self.name
        has_pa_zip = self.has_pa_zip
        pa_name = self.pa_name

        # filepath to store and/or retrieve cached data
        x_fname = os.path.join(DATA_DIR, name, 'x.pt')
        y_fname = os.path.join(DATA_DIR, name, 'y.pt')

        # if cached data exists, load it
        if os.path.exists(x_fname) and os.path.exists(y_fname):
            x = torch.load(x_fname)
            y = torch.load(y_fname)
            return x, y

        # fetch data points. always expects regular txt file
        x_url = f'http://cs.uef.fi/sipu/datasets/{name}.txt'
        with urllib.request.urlopen(x_url) as f:
            x_txt = f.read().decode('utf-8')

        # fetch ground truth, may be .zip or .txt format
        if has_pa_zip:
            y_url = f'http://cs.uef.fi/sipu/datasets/{pa_name}'
            with urllib.request.urlopen(y_url) as f1: # download zip file
                with tempfile.TemporaryFile() as f2: # store zip file in temp
                    f2.write(f1.read())
                    # the actual .pa file may have two different filenames,
                    # try them both
                    for y_path in ( f'{name}-ga.pa', f'{name}-label.pa' ):
                        try:
                            with ZipFile(f2) as zipobj: # unzips
                                with zipobj.open(y_path) as f3:
                                    y_txt = f3.read().decode('utf-8') # loads y
                        except KeyError as ex:
                            pass
                        break
        else:
            if pa_name is None: # if no .pa filename name given, use default
                y_url = f'http://cs.uef.fi/sipu/datasets/{name}.pa'
            else:
                y_url = f'http://cs.uef.fi/sipu/datasets/{pa_name}'
            with urllib.request.urlopen(y_url) as f:
                y_txt = f.read().decode('utf-8') # loads y

        # the .pa files are separated in two parts: the header and the labels.
        # both sections are separated by ------------
        # the following regex retrieves the labels only as string
        y_txt = re.match('^.*-+(.*)$', y_txt, re.S).group(1)

        # create directory to store cached results, if not exists
        try:
            os.mkdir(os.path.join(DATA_DIR, name))
        except FileExistsError as ex:
            pass

        # parse strings, store them in the datadir and return them
        x, y = parse_sipu(x_txt, y_txt)
        torch.save(x, x_fname)
        torch.save(y, y_fname)
        return x, y

# loads data from openml database
# https://www.openml.org/home
class OpenMLDataLoader(DataLoader):
    def load(self):
        x, y = sklearn.datasets.fetch_openml(self.name, return_X_y=True)
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.long))
        return x, y

# loads data from csv file, which is to be fetched from the internet
class CSVDataLoader(DataLoader):
    # name, preprocessor = same as super
    # url = the url to fetch the csv file
    # y_col = the column number of the labels (ground truth). if -1, it is
    #  assumed to be the last column
    # col_ignore = the indices of columns to ignore
    def __init__(self,
            name,
            url,
            preprocessor=lambda x: x,
            y_col=-1,
            col_ignore=[]):
        super().__init__(name, preprocessor=preprocessor)
        self.url = url
        self.y_col = y_col
        self.col_ignore = col_ignore

    def load(self):
        # retrieves cached data, if it exists
        x_fname = os.path.join(DATA_DIR, self.name, 'x.pt')
        y_fname = os.path.join(DATA_DIR, self.name, 'y.pt')
        if os.path.exists(x_fname) and os.path.exists(y_fname):
            x = torch.load(x_fname)
            y = torch.load(y_fname)
            return x, y

        # downloads csv
        with urllib.request.urlopen(self.url) as f1:
            df = pd.read_csv(f1)
        data = df.values

        # separates x and y from the data
        y = data[:,self.y_col]
        x = np.delete(data, self.y_col, axis=1)
        for col in self.col_ignore: # deletes columns to ignore
            x = np.delete(x, col, axis=1)

        # maps each unique value in y to a natural number id
        y_keys = dict(zip(np.unique(y), np.arange(np.unique(y).shape[0])))
        for i, lbl in enumerate(y):
            y[i] = y_keys[lbl]

        # converts to torch
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.long))

        # saves and returns data
        try:
            os.mkdir(os.path.join(DATA_DIR, self.name))
        except FileExistsError as ex:
            pass
        torch.save(x, x_fname)
        torch.save(y, y_fname)
        return x, y

# dataloaders to use
def get_dataloaders():
    url_template = 'https://ifcs.boku.ac.at/repository/data/{}/{}.csv'
    dataloaders = [ SipuDataLoader(f'dim{n}', preprocessor=GlobalScaler())
                    for n in ['032', '064', '128', '256', '512', '1024'] ] \
                + [ SipuDataLoader(
                        f'a{i}',
                        preprocessor=GlobalScaler(),
                        has_pa_zip=True,
                        pa_name='a-gt-pa.zip'
                    ) for i in range(1, 4) ] \
                + [ OpenMLDataLoader('mnist_784', preprocessor=Normalizer()),
                    OpenMLDataLoader('har', preprocessor=Normalizer()) ] \
                + [ CSVDataLoader(
                        'tetra',
                        url_template.format('tetragonula_bee', 'Tetragonula'),
                        preprocessor=GlobalScaler()
                    ),
                    CSVDataLoader(
                        'veronica',
                        url_template.format('veronica', 'Veronica'),
                        preprocessor=Normalizer(),
                        y_col=1,
                        col_ignore=[0]
                    ) ]
    return dataloaders
