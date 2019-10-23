import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = np.unique(y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def tune_classifier(classifier, params_to_tune, X, y):
    clf = GridSearchCV(classifier, params_to_tune, cv=5)
    clf.fit(X, y)
    print('Best parameters found')
    print(clf.best_params_)
    print('scores')
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
    print()
    return clf.best_params_

def normalize_by_year(X, col_header='YEAR', label_header='NBA_FINALS_APPEARANCE'):
    scaler = StandardScaler()
    dfs = []
    cols = list(X.columns)
    cols.remove(col_header)
    cols.remove(label_header)
    for grp in X.groupby(col_header):
        _, df = grp
        year = df[col_header].reset_index(drop=True)
        labels = df[label_header].reset_index(drop=True)
        df = df.drop([col_header, label_header], axis=1)
        normalized = scaler.fit_transform(df)
        normalized_df = pd.DataFrame(normalized, columns=cols)
        normalized_df[col_header] = year
        normalized_df[label_header] = labels
        dfs.append(normalized_df)

    return pd.concat(dfs)