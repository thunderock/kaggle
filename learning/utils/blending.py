
import glob
import pandas as pd
import numpy as np
from sklearn import metrics
from functools import partial
from scipy.optimize import fmin


class OptimizeAUC(object):

    def __init__(self):
        self.coef_ = 0

    def _auc(self, coef, X, y):
        x_coef = X * coef
        preds = np.sum(x_coef, axis=1)
        auc_score = metrics.roc_auc_score(y, preds)
        return -1.0 * auc_score

    def fit(self, X, y):
        partial_loss = partial(self._auc, X=X, y=y)
        init_coef = np.random.dirichlet(np.ones(X.shape[1]))
        self.coef_ = fmin(partial_loss, init_coef, disp=True)

    def predict(self, X):
        x_coef = X * self.coef_
        preds = np.sum(x_coef, axis=1)
        return preds

def run_training(pred_df, fold):
    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    test_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

    X_train = train_df[['lr_pred', 'cnt_pred', 'svd_rf_pred']].values
    X_test = test_df[['lr_pred', 'cnt_pred', 'svd_rf_pred']].values

    opt = OptimizeAUC()
    opt.fit(X_train, train_df.sentiment.values)
    preds = opt.predict(X_test)
    auc = metrics.roc_auc_score(test_df.sentiment.values, preds)
    print(f"{fold}, {auc}")
    test_df.loc[:, "opt_pred"] = preds
    return opt.coef_



if __name__ == "__main__":
    files = glob.glob("learning/utils/pred/*.csv")
    df = None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = df.merge(temp_df, on="id", how='left')



    targets = df.sentiment.values

    pred_cols = ['lr_pred', 'cnt_pred', 'svd_rf_pred']

    for col in pred_cols:
        auc = metrics.roc_auc_score(targets, df[col].values)
        print(f"{col}, overall auc = {auc}")


    print("average")
    avg_pred = np.mean(df[pred_cols].values, axis=1)
    print(metrics.roc_auc_score(targets, avg_pred))


    print("weighted average")
    lr_pred = df.lr_pred.values
    cnt_pred = df.cnt_pred.values
    svd_rf_pred = df.svd_rf_pred.values
    avg_pred = (lr_pred + 3 * cnt_pred + svd_rf_pred ) / 5
    print(metrics.roc_auc_score(targets, avg_pred))


    print("rank average")

    lr_pred = df.lr_pred.rank().values
    # print(len(set(lr_pred)), len(lr_pred))
    cnt_pred = df.cnt_pred.rank().values
    svd_rf_pred = df.svd_rf_pred.rank().values

    avg_pred = (lr_pred + cnt_pred + svd_rf_pred ) / 3
    # print(avg_pred)
    # print(targets)
    print(metrics.roc_auc_score(targets, avg_pred))



    print("weighted rank average")

    lr_pred = df.lr_pred.rank().values
    # print(len(set(lr_pred)), len(lr_pred))
    cnt_pred = df.cnt_pred.rank().values
    svd_rf_pred = df.svd_rf_pred.rank().values

    avg_pred = (lr_pred + 3 * cnt_pred + svd_rf_pred ) / 5
    # print(avg_pred)
    # print(targets)
    print(metrics.roc_auc_score(targets, avg_pred))


    print("# OPTIMIZE WEIGHTS")
    coefs = []

    for j in range(5):
        coefs.append(run_training(df, j))

    coefs = np.array(coefs)
    print(coefs)

    coefs = np.mean(coefs, axis=0)

    print(coefs)

    print("weighted average results")

    lr_pred = df.lr_pred.values
    cnt_pred = df.cnt_pred.values
    svd_rf_pred = df.svd_rf_pred.values
    wt_avg = (coefs[0] * lr_pred + coefs[1] * cnt_pred + coefs[2] * svd_rf_pred)

    print("optimal AUC")
    print(metrics.roc_auc_score(targets, wt_avg))
    # learng stacking here
