from utils import get_metrics_auc, \
     EarlyStopping
from load_data import load, remove_graph
import torch as th
import numpy as np
from model import Model
device = "cuda" if th.cuda.is_available() else "cpu"
def train_model(args , g , feature  , train_pos_idx , train_neg_idx , mask_train, label ) :
     # load model and optimizer
    num_nodes = sum([g.num_nodes(nt) for nt in g.ntypes])
    model = Model(etypes=g.etypes, ntypes=g.ntypes,
                    in_feats=feature['drug'].shape[1],
                    num_nodes=num_nodes,
                    args=args
                    )
    model.to(device)

    optimizer = th.optim.Adam(model.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    optim_scheduler = th.optim.lr_scheduler.CyclicLR(optimizer,
                                                        base_lr=0.1 * args.learning_rate,
                                                        max_lr=args.learning_rate,
                                                        gamma=0.995,
                                                        step_size_up=20,
                                                        mode="exp_range",
                                                        cycle_momentum=False)
    criterion = th.nn.BCEWithLogitsLoss(pos_weight=th.tensor(len(train_neg_idx[0]) / len(train_pos_idx[0])))
    print('Loss pos weight: {:.3f}'.format(len(train_neg_idx[0]) / len(train_pos_idx[0])))
    stopper = EarlyStopping(patience=args.patience, saved_path=args.saved_path)

    # model training
    for epoch in range(1, args.epoch + 1):
        model.train()
        score = model(g, feature)
        pred = th.sigmoid(score)
        loss = criterion(score[mask_train].cpu().flatten(),
                            label[mask_train].cpu().flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optim_scheduler.step()
        model.eval()
        AUC, _ = get_metrics_auc(label[mask_train].cpu().detach().numpy(),
                                    pred[mask_train].cpu().detach().numpy())
        early_stop = stopper.step(loss.item(), AUC, model)

        # if epoch % 1 == 0:
        print('Epoch {} Loss: {:.3f}; Train AUC {:.3f}'.format(epoch, loss.item(), AUC))
        print('-' * 40)
        if early_stop:
            break
def train_cv(args , fold ,data ,df, data_pos , data_neg , train_pos_idx , test_pos_idx ,train_neg_idx , test_neg_idx ) :
    print('{}-Cross Validation: Fold {}'.format(args.nfold, fold))

    # get the index list for train and test set
    train_pos_id, test_pos_id = data_pos[train_pos_idx], data_pos[test_pos_idx]
    train_neg_id, test_neg_id = data_neg[train_neg_idx], data_neg[test_neg_idx]
    train_pos_idx = [tuple(train_pos_id[:, 0]), tuple(train_pos_id[:, 1])]
    test_pos_idx = [tuple(test_pos_id[:, 0]), tuple(test_pos_id[:, 1])]
    train_neg_idx = [tuple(train_neg_id[:, 0]), tuple(train_neg_id[:, 1])]
    test_neg_idx = [tuple(test_neg_id[:, 0]), tuple(test_neg_id[:, 1])]
    assert len(test_pos_idx[0]) + len(test_neg_idx[0]) + len(train_pos_idx[0]) + len(train_neg_idx[0]) == len(data)

    g = load(args.dataset)
    print(g)
    # remove test set DDA from train graph
    g = remove_graph(g, test_pos_id[:, :-1]).to(device)
    if args.dataset == 'Kdataset':
        feature = {'drug': g.nodes['drug'].data['h'],
                    'disease': g.nodes['disease'].data['h'],
                    'protein': g.nodes['protein'].data['h'],
                    'gene': g.nodes['gene'].data['h'],
                    'pathway': g.nodes['pathway'].data['h']}
    elif args.dataset == 'Bdataset':
        feature = {'drug': g.nodes['drug'].data['h'],
                    'disease': g.nodes['disease'].data['h'],
                    'protein': g.nodes['protein'].data['h']}
    else:
        feature = {'drug': g.nodes['drug'].data['h'],
                    'disease': g.nodes['disease'].data['h']}

    # get the mask list for train and test set that used for performance calculation
    mask_label = np.ones(df.shape)
    mask_label[test_pos_idx[0], test_pos_idx[1]] = 0
    mask_label[test_neg_idx[0], test_neg_idx[1]] = 0
    mask_test = np.where(mask_label == 0)
    mask_test = [tuple(mask_test[0]), tuple(mask_test[1])]
    mask_train = np.where(mask_label == 1)
    mask_train = [tuple(mask_train[0]), tuple(mask_train[1])]

    print('Number of total training samples: {}, pos samples: {}, neg samples: {}'.format(len(mask_train[0]),
                                                                                            len(train_pos_idx[0]),
                                                                                            len(train_neg_idx[0])))
    print('Number of total testing samples: {}, pos samples: {}, neg samples: {}'.format(len(mask_test[0]),
                                                                                            len(test_pos_idx[0]),
                                                                                            len(test_neg_idx[0])))
    label = th.tensor(df).float().to(device)
    
    train_model(args , g , feature  , train_pos_idx , train_neg_idx , mask_train, label )


def test_cv(args ,dir,df,fold,pred_result ,data_pos , train_pos_idx ,test_pos_idx  ,data_neg ,train_neg_idx ,test_neg_idx ) :
      # get the index list for test set
    _, test_pos_id = data_pos[train_pos_idx], data_pos[test_pos_idx]
    _, test_neg_id = data_neg[train_neg_idx], data_neg[test_neg_idx]
    test_pos_idx = [tuple(test_pos_id[:, 0]), tuple(test_pos_id[:, 1])]
    test_neg_idx = [tuple(test_neg_id[:, 0]), tuple(test_neg_id[:, 1])]

    g = load(args.dataset)
    g = remove_graph(g, test_pos_id[:, :-1]).to(device)
    if args.dataset == 'Kdataset':
        feature = {'drug': g.nodes['drug'].data['h'],
                    'disease': g.nodes['disease'].data['h'],
                    'protein': g.nodes['protein'].data['h'],
                    'gene': g.nodes['gene'].data['h'],
                    'pathway': g.nodes['pathway'].data['h']}
    elif args.dataset == 'Bdataset':
        feature = {'drug': g.nodes['drug'].data['h'],
                    'disease': g.nodes['disease'].data['h'],
                    'protein': g.nodes['protein'].data['h']}
    else:
        feature = {'drug': g.nodes['drug'].data['h'],
                    'disease': g.nodes['disease'].data['h']}

    # get the mask list for test set that used for performance calculation
    mask_label = np.ones(df.shape)
    mask_label[test_pos_idx[0], test_pos_idx[1]] = 0
    mask_label[test_neg_idx[0], test_neg_idx[1]] = 0
    mask_test = np.where(mask_label == 0)
    mask_test = [tuple(mask_test[0]), tuple(mask_test[1])]

    assert len(mask_test[0]) == len(test_neg_idx[0]) + len(test_pos_idx[0])
    label = th.tensor(df).float().to(device)

    
    num_nodes = sum([g.num_nodes(nt) for nt in g.ntypes])
    model = Model(etypes=g.etypes, ntypes=g.ntypes,
                    in_feats=feature['drug'].shape[1],
                    num_nodes=num_nodes,
                    args=args
                    )
    model.to(device)
    # Try to load matching checkpoint; fall back to non-strict load or skip if incompatible
    checkpoint = th.load(dir[fold])
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as e:
        print(f"Warning: checkpoint and model state_dict mismatch: {e}")
        try:
            model.load_state_dict(checkpoint, strict=False)
            print('Loaded checkpoint with strict=False (some keys ignored)')
        except Exception as e2:
            print('Failed to load checkpoint; continuing without loading. Error:', e2)
    model.eval()
    pred = th.sigmoid(model(g, feature))
    AUC, AUPR = get_metrics_auc(label[mask_test].cpu().detach().numpy(), pred[mask_test].cpu().detach().numpy())
    pred = pred.cpu().detach().numpy()
    pred_result[test_pos_idx[0], test_pos_idx[1]] = pred[test_pos_idx[0], test_pos_idx[1]]
    pred_result[test_neg_idx[0], test_neg_idx[1]] = pred[test_neg_idx[0], test_neg_idx[1]]   
    print('Fold {} Test AUC {:.3f}; AUPR: {:.3f}'.format(fold, AUC, AUPR))
    return label