import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

base_path = '/home/ashutosh/code/kaggle/RFSC_audio_detection/'
train = pd.read_csv(os.path.join(base_path, "data/train_tp.csv")).sort_values("recording_id")
ss = pd.read_csv(os.path.join(base_path, "data/sample_submission.csv"))

FOLDS = 5
SEED = 42

train_gby = train.groupby("recording_id")[["species_id"]].first().reset_index()
train_gby = train_gby.sample(frac=1, random_state=SEED).reset_index(drop=True)
train_gby.loc[:, 'kfold'] = -1

X = train_gby["recording_id"].values
y = train_gby["species_id"].values

kfold = StratifiedKFold(n_splits=FOLDS)
for fold, (t_idx, v_idx) in enumerate(kfold.split(X, y)):
    train_gby.loc[v_idx, "kfold"] = fold

train = train.merge(train_gby[['recording_id', 'kfold']], on="recording_id", how="left")
print(train.kfold.value_counts())
train.to_csv(os.path.join(base_path, "train_folds.csv"), index=False)
