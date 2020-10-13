import imblearn
import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(n_classes=2, class_sep=1.5, weights=[.9, .1],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1, n_samples=100, random_state=32)

df = pd.DataFrame(X)
df['target'] = y

df.target.value_counts().plot(kind='bar', title='Count (target)');
