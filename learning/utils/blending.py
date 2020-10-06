
import glob
import pandas as pd

from sklearn import metrics

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
        
