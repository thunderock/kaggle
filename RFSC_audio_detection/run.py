import warnings
warnings.filterwarnings('ignore')

import os, time, librosa, random
import numpy as np, pandas as pd

import torch, torch.nn as nn
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import datasets
import models
import losses
import functions
import augmentations

class args:
    DEBUG = True
    exp_name = "resnest50d_5fold_base"
    network = "AudioClassifier"
    pretrain_weights = None
    model_param = {
        'encoder' : 'resnest50d',
        'sample_rate': 48000,
        'window_size' : 512 * 2,
        'hop_size' : 345 * 2,
        'mel_bins' : 128,
        'fmin' : 20,
        'fmax' : 48000 // 2,
        'classes_num' : 24
    }
    losses = "BCEWithLogitsLoss"
    lr = 1e-3
    step_scheduler = True
    epoch_scheduler = False
    period = 30
    seed = 42
    start_epoch = 0
    epochs = 50
    batch_size = 16
    num_workers = 2
    early_stop = 10

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    train_csv = "train_folds.csv"
    test_csv = "test_df.csv"
    sub_csv = "data/sample_submission.csv"
    output_dir = "weights"

def main(fold):

    # Setting seed
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    args.fold = fold
    args.save_path = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(args.save_path, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)

    sub_df = pd.read_csv(args.sub_csv)
    if args.DEBUG:
        train_df = train_df.sample(100)
        sub_df = sub_df.sample(10)
        args.batch_size = 4
        args.epochs = 5
    train_fold = train_df[train_df.kfold != fold]
    valid_fold = train_df[train_df.kfold == fold]

    train_dataset = datasets.AudioDataset(
        df=train_fold,
        period=args.period,
        transforms=augmentations.augmenter,
        train=True,
        data_path="data/train"
    )
    valid_dataset = datasets.AudioDataset(
        df=valid_fold,
        period=args.period,
        transforms=None,
        train=True,
        data_path="data/train"
    )

    test_dataset = datasets.TestDataset(
        df=sub_df,
        period=args.period,
        transforms=None,
        train=False,
        data_path="data/test"
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size//2,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    model = models.__dict__[args.network](**args.model_param)
    model = model.to(args.device)

    if args.pretrain_weights:
        print("---------------------loading pretrain weights")
        model.load_state_dict(torch.load(args.pretrain_weights, map_location=args.device)["model"], strict=False)
        model = model.to(args.device)

    criterion = losses.__dict__[args.losses]()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_train_steps = int(len(train_loader) * args.epochs)
    num_warmup_steps = int(0.1 * args.epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

    best_lwlrap = -np.inf
    for epoch in range(args.start_epoch, args.epochs):
        train_avg, train_loss = functions.train_epoch(args, model, train_loader, criterion, optimizer, scheduler, epoch)
        valid_avg, valid_loss = functions.valid_epoch(args, model, valid_loader, criterion, epoch)

        if args.epoch_scheduler:
            scheduler.step()

        content = f"""
                {time.ctime()} \n
                Fold:{args.fold}, Epoch:{epoch}, lr:{optimizer.param_groups[0]['lr']:.7}\n
                Train Loss:{train_loss:0.4f} - LWLRAP:{train_avg['lwlrap']:0.4f}\n
                Valid Loss:{valid_loss:0.4f} - LWLRAP:{valid_avg['lwlrap']:0.4f}\n
        """
        print(content)
        with open(f'{args.save_path}/log_{args.exp_name}.txt', 'a') as appender:
            appender.write(content+'\n')

        if valid_avg['lwlrap'] > best_lwlrap:
            print(f"########## >>>>>>>> Model Improved From {best_lwlrap} ----> {valid_avg['lwlrap']}")
            torch.save(model.state_dict(), os.path.join(args.save_path, f'fold-{args.fold}.bin'))
            best_lwlrap = valid_avg['lwlrap']
        #torch.save(model.state_dict(), os.path.join(args.save_path, f'fold-{args.fold}_last.bin'))


    model.load_state_dict(torch.load(os.path.join(args.save_path, f'fold-{args.fold}.bin'), map_location=args.device))
    model = model.to(args.device)

    target_cols = sub_df.columns[1:].values.tolist()
    test_pred, ids = functions.test_epoch(args, model, test_loader)
    print(np.array(test_pred).shape)

    test_pred_df = pd.DataFrame({
        "recording_id" : sub_df.recording_id.values
    })
    # need to fix this line
    test_pred_df[target_cols] = np.array(test_pred)
    test_pred_df.to_csv(os.path.join(args.save_path, f"fold-{args.fold}-submission.csv"), index=False)
    print(os.path.join(args.save_path, f"fold-{args.fold}-submission.csv"))




if __name__ == "__main__":
    for fold in range(5):
        if fold == 0:
            main(fold)
