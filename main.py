import os
import glob
import numpy as np
import pandas as pd
from warnings import simplefilter
from sklearn.model_selection import KFold
from utils import  set_seed, plot_result_auc, \
    plot_result_aupr, get_metrics
from args import args
import scipy.io as sio
from train import train_cv ,test_cv

def train():
    simplefilter(action='ignore', category=FutureWarning)
    print(args)
    try:
        os.mkdir(args.saved_path)
    except:
        pass

    # load DDA data for Kfold splitting
    if args.dataset in ['Kdataset', 'Bdataset']:
        df = pd.read_csv('./dataset/{}/{}_baseline.csv'.format(args.dataset, args.dataset), header=None).values
    else:
        m = sio.loadmat('./dataset/Cdataset/Cdataset.mat')
        df = m['didr'].T
    data = np.array([[i, j, df[i, j]] for i in range(df.shape[0]) for j in range(df.shape[1])])
    print(np.array(np.where(data[:, -1] == 1)).shape)
    data = data.astype('int64')
    data_pos = data[np.where(data[:, -1] == 1)[0]]
    data_neg = data[np.where(data[:, -1] == 0)[0]]
    assert len(data) == len(data_pos) + len(data_neg)

    #------------training-------------
    set_seed(args.seed)
    kf = KFold(n_splits=args.nfold, shuffle=True, random_state=args.seed)
    fold = 0
    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kf.split(data_pos), kf.split(data_neg)):
        fold += 1
        train_cv(args , fold ,data ,df, data_pos , data_neg , train_pos_idx , test_pos_idx ,train_neg_idx , test_neg_idx )
        
    #------------testing--------------
    fold = 0
    dir = glob.glob(args.saved_path + '/*.pth')
    pred_result = np.zeros(df.shape)
    print('model testing')
    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kf.split(data_pos),
                                                                            kf.split(data_neg)):
        label = test_cv(args ,dir,df,fold,pred_result ,data_pos , train_pos_idx ,test_pos_idx  ,data_neg ,train_neg_idx ,test_neg_idx )
        fold += 1


    #---------save the result-------------
    AUC, aupr, acc, f1, pre, rec, spe = get_metrics(label.cpu().detach().numpy().flatten(), pred_result.flatten())
    print(
        'Overall: AUC {:.3f}; AUPR: {:.3f}; Acc: {:.3f}; F1: {:.3f}; Precision {:.3f}; Recall {:.3f}; Specificity {:.3F}'.
            format(AUC, aupr, acc, f1, pre, rec, spe))
    pd.DataFrame(pred_result).to_csv(os.path.join(args.saved_path, 'result.csv'), index=False, header=False)
    plot_result_auc(args, data[:, -1].flatten(), pred_result.flatten(), AUC)
    plot_result_aupr(args, data[:, -1].flatten(), pred_result.flatten(), aupr)


if __name__ == '__main__':
    train()
