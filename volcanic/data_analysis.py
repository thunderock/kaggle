import os
import gc
import numpy as np
import pandas as pd

from time import time
from time import ctime

import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

import joblib
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()-1

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold

from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features
from tsfresh.feature_extraction import settings, extract_features, MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters


def features_generator(path_to_file):
    signals = pd.read_csv(path_to_file)
    seg = int(path_to_file.split('/')[-1].split('.')[0])
    signals['segment_id'] = seg
    
    sel = signals.fillna(0).astype(bool).sum(axis=0) / 60001 > 0.5
    signals = signals.fillna(0).loc[:,sel]

    extracted_features = extract_features(signals.iloc[:,:], 
                                          column_id = 'segment_id', 
                                          default_fc_parameters=EfficientFCParameters(),
                                          n_jobs = 0,
                                          disable_progressbar = True,
                                          chunksize = None,
                                         )
    return extracted_features


train_path_to_signals = 'data/predict-volcanic-eruptions-ingv-oe/train/'
train_files_list = [os.path.join(train_path_to_signals, file) for file in os.listdir(train_path_to_signals)]
rows = Parallel(n_jobs=6)(delayed(features_generator)(ex) for ex in tqdm(train_files_list[:]))  
train_set = pd.concat(rows, axis=0)

test_path_to_signals = 'data/predict-volcanic-eruptions-ingv-oe/test/'
test_files_list = [os.path.join(test_path_to_signals, file) for file in os.listdir(test_path_to_signals)]
rows = Parallel(n_jobs=6)(delayed(features_generator)(ex) for ex in tqdm(test_files_list[:]))  
test_set = pd.concat(rows, axis=0)

print(train_set.head(10))



print(test_set.head(10))


train_set.to_csv('train_features.csv')


test_set.to_csv('test_features.csv')




