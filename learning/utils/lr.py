import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer


def run_training(fold):
    df = pd.read_csv("learning/utils/train_folds.csv")
    df.review = df.review.apply(str)


    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    tfv = TfidfVectorizer()
    tfv.fit(df_train.review.values)

    X_train = tfv.transform(df_train.review.values)
    X_test = tfv.transform(df_valid.review.values)

    y_train = df_train.sentiment.values
    y_test = df_valid.sentiment.values

    clf = linear_model.LogisticRegression()
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_test)[:, 1]

    auc = metrics.roc_auc_score(y_test, pred)
    print(f"fold={fold}, auc={auc}")
    df_valid.loc[:, "lr_pred"] = pred

    return df_valid[["id", "sentiment", "kfold", "lr_pred"]]



dfs = []
for i in range(5):
    temp_df = run_training(i)
    dfs.append(temp_df)

final_valid_df = pd.concat(dfs)
print(final_valid_df.shape)
final_valid_df.to_csv("learning/utils/pred_lr.csv", index=False)
