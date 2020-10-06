import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import decomposition
from sklearn import ensemble
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings
warnings.filterwarnings("ignore")



def run_training_svd(fold, vectorizer):
    df = pd.read_csv("learning/utils/train_folds.csv")
    df.review = df.review.apply(str)


    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)


    vectorizer.fit(df_train.review.values)

    X_train = vectorizer.transform(df_train.review.values)
    X_test = vectorizer.transform(df_valid.review.values)

    svd  = decomposition.TruncatedSVD(n_components=120)
    svd.fit(X_train)
    X_train_svd = svd.transform(X_train)
    X_test_svd = svd.transform(X_test)

    y_train = df_train.sentiment.values
    y_test = df_valid.sentiment.values

    clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100)
    clf.fit(X_train_svd, y_train)
    pred = clf.predict_proba(X_test_svd)[:, 1]

    auc = metrics.roc_auc_score(y_test, pred)
    print(f"fold={fold}, auc={auc}")
    df_valid.loc[:, "svd_rf_pred"] = pred

    return df_valid[["id", "sentiment", "kfold", "svd_rf_pred"]]


def run_training(fold, vectorizer, name):
    df = pd.read_csv("learning/utils/train_folds.csv")
    df.review = df.review.apply(str)


    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)


    vectorizer.fit(df_train.review.values)

    X_train = vectorizer.transform(df_train.review.values)
    X_test = vectorizer.transform(df_valid.review.values)

    y_train = df_train.sentiment.values
    y_test = df_valid.sentiment.values

    clf = linear_model.LogisticRegression()
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_test)[:, 1]

    auc = metrics.roc_auc_score(y_test, pred)
    print(f"fold={fold}, auc={auc}")
    df_valid.loc[:, name] = pred

    return df_valid[["id", "sentiment", "kfold", name]]



dfs = []
for i in range(5):
    v = TfidfVectorizer()
    temp_df = run_training_svd(i, v)
    dfs.append(temp_df)

final_valid_df = pd.concat(dfs)
print(final_valid_df.shape)
final_valid_df.to_csv("learning/utils/pred/pred_lr_svd.csv", index=False)



dfs = []
for i in range(5):
    v = TfidfVectorizer()
    temp_df = run_training(i, v, "lr_pred")
    dfs.append(temp_df)

final_valid_df = pd.concat(dfs)
print(final_valid_df.shape)
final_valid_df.to_csv("learning/utils/pred/pred_lr_tfidf.csv", index=False)



dfs = []
for i in range(5):
    v = CountVectorizer()
    temp_df = run_training(i, v, "cnt_pred")
    dfs.append(temp_df)

final_valid_df = pd.concat(dfs)
print(final_valid_df.shape)
final_valid_df.to_csv("learning/utils/pred/pred_lr_count.csv", index=False)
