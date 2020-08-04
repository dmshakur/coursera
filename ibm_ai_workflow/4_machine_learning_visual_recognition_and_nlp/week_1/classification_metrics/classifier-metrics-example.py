#!/usr/bin/env python

"""
simple confusion matrix example

plot inspired by 
https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
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
    classes = classes[np.sort(np.unique(np.union1d(y_true, y_pred)))]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    if normalize:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0,vmax=1.0)
    else:
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

    ax.set_ylim((-0.5,len(classes)-0.5))
    
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
    return(ax)


if __name__ == "__main__":

    y_true = [0,1,0,2,1,0,0,2,1,1,0,1]
    y_pred = [0,0,0,2,1,0,2,1,2,1,0,1]
    target_names = np.array(['retained', 'unretained', 'on hold'])
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # Plot normalized confusion matrix
    plot_confusion_matrix(y_true, y_pred,classes=target_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
