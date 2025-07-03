import pandas as pd
import os
import sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, randint, loguniform

os.chdir('/Users/saksh/Documents/PhD_Nice/Manuscripts/CEFIPRA_manuscript/Tables')
data = pd.read_csv('Final_1011_AFR_phenotypes.csv')
df_phen = data.iloc[:, 11:]

imputer = KNNImputer(n_neighbors=5)
df_phen = imputer.fit_transform(df_phen)
df_phen = pd.DataFrame(df_phen, columns=data.columns[11:])

#for col in df_phen.columns:
 #   df_phen[col + '_cat'] = pd.qcut(df_phen[col], q=3, labels=['low','med','high'])

def categorical_distinction(x):
    if x < -1:
        return 'low'
    elif -1 <= x <= 1:
        return 'med'
    else:
        return 'high'


def quartile_labels(series):
    """
    Divide the data range into 4 equal-width quartiles and label each value accordingly.
    Labels: 'Q1', 'Q2', 'Q3', 'Q4'
    """
    min_val = series.min()
    max_val = series.max()
    bins = np.linspace(min_val, max_val, 5)  # 4 intervals -> 5 edges
    labels = ['Q1', 'Q2', 'Q3', 'Q4']

    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)

def apply_quartiles_to_df(df):
    return df.apply(quartile_labels)


df_labeled = apply_quartiles_to_df(df_phen)

#df_labeled = df_phen.applymap(categorical_distinction)

df_labeled.index = list(data.iloc[:, 1])

PAF_matrix = pd.read_csv('PAF_with_popstruc_1011_gen.csv', index_col=0)
PhenCategorical = df_labeled.reindex(PAF_matrix.index)

X = PAF_matrix
y = df_labeled.iloc[:,1]
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


"""
model = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100,
                      objective='binary:logistic', base_score=0.5, eval_metric='logloss')

model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

"""
GBM_distributions = dict( learning_rate=uniform(0.1, 1),
                         max_depth=randint(2, 10), subsample=uniform(0, 1),
                         min_samples_split=randint(2, 100), min_samples_leaf=randint(2, 100),
                         n_estimators=randint(4, 100), criterion=['friedman_mse', 'squared_error'])

GBM_training = RandomizedSearchCV(GradientBoostingClassifier(),
                                  GBM_distributions, n_iter=10, verbose=10,
                                  cv=5).fit(X_train, y_train)

modelGBM = GBM_training.best_estimator_
y_train_predGBM = modelGBM.predict(X_train)
y_test_predGBM = modelGBM.predict(X_test)

print("Accuracy test:", accuracy_score(y_test, y_test_predGBM))
print("Accuracy train:", accuracy_score(y_train, y_train_predGBM))
cm_train = confusion_matrix(y_train, y_train_predGBM, labels=np.unique(y))
cm_test = confusion_matrix(y_test, y_test_predGBM, labels=np.unique(y))
plt.figure(figsize=(6, 5))
sns.heatmap(cm_train,annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Training, 0.9)')
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Testing, 0.1)')
plt.show()


