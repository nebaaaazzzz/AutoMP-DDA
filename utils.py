import datetime
import numpy as np
import torch
import random
import seaborn
import os
from sklearn.metrics import roc_curve, roc_auc_score, \
    precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


def get_metrics_auc(real_score, predict_score):
    AUC = roc_auc_score(real_score, predict_score)
    AUPR = average_precision_score(real_score, predict_score)
    return AUC, AUPR


def get_metrics(real_score, predict_score):
    """Memory-efficient calculation of performance metrics.

    Uses sklearn's roc_auc_score and average_precision_score for AUC/AUPR, and
    precision_recall_curve to find the threshold that maximizes F1, then computes
    Accuracy and Specificity at that threshold.

    Returns
    -------
    AUC, AUPR, Accuracy, F1-Score, Precision, Recall, Specificity
    """
    # ensure inputs are numpy arrays
    real = np.array(real_score).flatten()
    pred = np.array(predict_score).flatten()

    # AUC and AUPR (these operate directly on arrays)
    try:
        AUC = roc_auc_score(real, pred)
    except Exception:
        AUC = float('nan')
    try:
        AUPR = average_precision_score(real, pred)
    except Exception:
        AUPR = float('nan')

    # Use precision-recall curve to compute F1 for thresholds
    precision, recall, thresholds = precision_recall_curve(real, pred)
    # precision and recall have length = len(thresholds) + 1
    if len(thresholds) == 0:
        # degenerate case: no threshold found (all scores identical)
        best_thresh = 0.5
        prec = precision[-1]
        rec = recall[-1]
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
    else:
        # compute F1 for each threshold (align precision/recall appropriately)
        f1_list = 2 * (precision[1:] * recall[1:]) / (precision[1:] + recall[1:] + 1e-12)
        best_idx = int(np.nanargmax(f1_list))
        f1 = float(f1_list[best_idx])
        prec = float(precision[best_idx + 1])
        rec = float(recall[best_idx + 1])
        best_thresh = float(thresholds[best_idx])

    # compute confusion matrix at best_thresh
    pred_bin = (pred >= best_thresh).astype(int)
    tp = int(np.sum((pred_bin == 1) & (real == 1)))
    fp = int(np.sum((pred_bin == 1) & (real == 0)))
    tn = int(np.sum((pred_bin == 0) & (real == 0)))
    fn = int(np.sum((pred_bin == 0) & (real == 1)))

    accuracy = (tp + tn) / (len(real) + 1e-12)
    specificity = tn / (tn + fp + 1e-12)

    return AUC, AUPR, accuracy, f1, prec, rec, specificity


def set_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class EarlyStopping(object):
    def __init__(self, patience=10, saved_path='.'):
        dt = datetime.datetime.now()
        self.filename = os.path.join(saved_path, 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second))
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)


def plot_result_auc(args, label, predict, auc):
    """Plot the ROC curve for predictions.
    Parameters
    ----------
    args: argumentation
    label: true labels
    predict: model predictions
    auc: calculated AUROC score
    """
    seaborn.set_style()
    fpr, tpr, threshold = roc_curve(label, predict)
    plt.figure()
    lw = 2
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(os.path.join(args.saved_path, 'result_auc.png'))
    plt.clf()


def plot_result_aupr(args, label, predict, aupr):
    """Plot the ROC curve for predictions.
    Parameters
    ----------
    args: argumentation
    label: true labels
    predict: model predictions
    aupr: calculated AUPR score
    """
    seaborn.set_style()
    precision, recall, thresholds = precision_recall_curve(label, predict)
    plt.figure()
    lw = 2
    plt.figure(figsize=(8, 8))
    plt.plot(precision, recall, color='darkorange',
             lw=lw, label='AUPR Score (area = %0.4f)' % aupr)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('RPrecision/Recall Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(os.path.join(args.saved_path, 'result_aupr.png'))
    plt.clf()
