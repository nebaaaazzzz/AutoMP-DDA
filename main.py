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

os.environ.DGLBACKEND = "pytorch"

def train():
    simplefilter(action='ignore', category=FutureWarning)
    print(args)
    os.mkdir(args.saved_path)

    # load DDA data for Kfold splitting
    if args.dataset in ['Kdataset', 'Bdataset']:
        df = pd.read_csv('./dataset/{}/{}_baseline.csv'.format(args.dataset, args.dataset), header=None).values
    elif args.dataset == 'Fdataset' : 
        m = sio.loadmat('./dataset/Fdataset/Fdataset.mat')
        df = m['didr'].T
    elif args.dataset =='Cdataset' :
        m = sio.loadmat('./dataset/Cdataset/Cdataset.mat')
        df = m['didr'].T
    else:
        raise NameError()
    
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
        train_cv(args  ,fold,data ,df, data_pos , data_neg , train_pos_idx , test_pos_idx ,train_neg_idx , test_neg_idx )
        
    #------------testing--------------
    fold = 1  # fold numbering starts at 1 to match training saved checkpoint tags
    dir = glob.glob(args.saved_path + '/*.pth')
    pred_result = np.zeros(df.shape)
    print('model testing')
    per_fold_metrics = []
    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kf.split(data_pos),
                                                                            kf.split(data_neg)):
        label, metrics = test_cv(args ,dir,df,fold,pred_result ,data_pos , train_pos_idx ,test_pos_idx  ,data_neg ,train_neg_idx ,test_neg_idx )
        per_fold_metrics.append(metrics)
        fold += 1

    #---------save the result-------------
    AUC, aupr, acc, f1, pre, rec, spe = get_metrics(label.cpu().detach().numpy().flatten(), pred_result.flatten())
    print(
        'Overall: AUC {:.3f}; AUPR: {:.3f}; Acc: {:.3f}; F1: {:.3f}; Precision {:.3f}; Recall {:.3f}; Specificity {:.3F}'.
            format(AUC, aupr, acc, f1, pre, rec, spe))
    pd.DataFrame(pred_result).to_csv(os.path.join(args.saved_path, 'result.csv'), index=False, header=False)
    plot_result_auc(args, data[:, -1].flatten(), pred_result.flatten(), AUC)
    plot_result_aupr(args, data[:, -1].flatten(), pred_result.flatten(), aupr)

    # Save per-fold metrics and compute mean/std
    cols = ['AUC', 'AUPR', 'Acc', 'F1', 'Precision', 'Recall', 'Specificity']
    try:
        per_df = pd.DataFrame(per_fold_metrics, columns=cols)
        per_df.index = range(1, len(per_df) + 1)  # fold numbering starts at 1
        per_df.to_csv(os.path.join(args.saved_path, 'metrics_per_fold.csv'), index_label='fold')
        summary = pd.DataFrame([per_df.mean(), per_df.std()], index=['mean', 'std'])
        summary.to_csv(os.path.join(args.saved_path, 'metrics_summary.csv'))
        # human readable summary
        with open(os.path.join(args.saved_path, 'metrics_summary.txt'), 'w') as f:
            f.write('Per-fold metrics:\n')
            f.write(per_df.to_string())
            f.write('\n\nSummary (mean ± std):\n')
            for c in cols:
                f.write(f"{c}: {summary.loc['mean', c]:.4f} ± {summary.loc['std', c]:.4f}\n")
        print('Saved per-fold metrics to', os.path.join(args.saved_path, 'metrics_per_fold.csv'))
        print('Saved metrics summary to', os.path.join(args.saved_path, 'metrics_summary.csv'))

        # Save best-performing fold model (by AUC)
        try:
            import shutil
            best_fold = per_df['AUC'].idxmax()  # index matches fold numbering (1-based)
            best_auc = per_df.loc[best_fold, 'AUC']
            # find the checkpoint with matching fold tag
            pths = glob.glob(os.path.join(args.saved_path, '*.pth'))
            best_candidates = [p for p in pths if f'_fold{best_fold}' in os.path.basename(p)]
            if best_candidates:
                best_ckpt = best_candidates[0]
            else:
                # fallback: pick the earliest matching the fold order
                pths_sorted = sorted(pths)
                if len(pths_sorted) >= best_fold:
                    best_ckpt = pths_sorted[best_fold-1]
                    print('Warning: could not find fold tag in filenames; using', best_ckpt)
                else:
                    raise FileNotFoundError('No checkpoint found for best fold {}'.format(best_fold))
            # copy to best model filename
            best_name = os.path.join(args.saved_path, f'best_model_fold{best_fold}_AUC{best_auc:.4f}.pth')
            shutil.copy(best_ckpt, best_name)
            # also keep a stable name
            shutil.copy(best_ckpt, os.path.join(args.saved_path, 'best_model.pth'))
            print(f'Saved best model for fold {best_fold} (AUC={best_auc:.4f}) to {best_name}')
        except Exception as e:
            print('Warning: failed to save best-performing model:', e)
    except Exception as e:
        print('Warning: failed to save per-fold metrics:', e)


if __name__ == '__main__':
    train()
