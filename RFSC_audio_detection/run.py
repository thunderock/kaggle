import warnings
warnings.filterwarnings('ignore')

import os
import time
import random
import numpy as np
import pandas as pd

import torch
from transformers import get_linear_schedule_with_warmup
import datasets
import models
import losses
import functions
import augmentations
from tqdm import tqdm

class Args:
    DEBUG = True
    exp_name = "resnest50d_5fold_base"
    network = "AudioClassifier"
    pretrain_weights = None
    model_param = {
        'encoder': 'resnest50d',
        'sample_rate': 48000,
        'window_size': 512 * 2,
        'hop_size': 345 * 2,
        'mel_bins': 128,
        'fmin': 20,
        'fmax': 48000 // 2,
        'classes_num': 24
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
    train_csv = "/home/ashutosh/code/kaggle/RFSC_audio_detection/train_folds.csv"
    test_csv = "test_df.csv"
    sub_csv = "/home/ashutosh/code/kaggle/RFSC_audio_detection/data/sample_submission.csv"
    output_dir = "/home/ashutosh/code/kaggle/RFSC_audio_detection/weights"


def main(fold):
    seed = Args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    Args.fold = fold
    Args.save_path = os.path.join(Args.output_dir, Args.exp_name)
    os.makedirs(Args.save_path, exist_ok=True)

    train_df = pd.read_csv(Args.train_csv)

    sub_df = pd.read_csv(Args.sub_csv)
    if Args.DEBUG:
        train_df = train_df.sample(100)
        sub_df = sub_df.sample(10)
        Args.batch_size = 4
        Args.epochs = 5
    train_fold = train_df[train_df.kfold != fold]
    valid_fold = train_df[train_df.kfold == fold]

    train_dataset = datasets.AudioDataset(
        df=train_fold,
        period=Args.period,
        transforms=augmentations.augmenter,
        train=True,
        data_path="data/train"
    )
    valid_dataset = datasets.AudioDataset(
        df=valid_fold,
        period=Args.period,
        transforms=None,
        train=True,
        data_path="data/train"
    )

    test_dataset = datasets.TestDataset(
        df=sub_df,
        period=Args.period,
        transforms=None,
        train=False,
        data_path="data/test"
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=Args.num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=Args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=Args.num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Args.batch_size//2,
        shuffle=False,
        drop_last=False,
        num_workers=Args.num_workers
    )

    model = models.__dict__[Args.network](**Args.model_param)
    model = model.to(Args.device)

    if Args.pretrain_weights:
        print("---------------------loading pretrain weights")
        model.load_state_dict(torch.load(Args.pretrain_weights, map_location=Args.device)["model"], strict=False)
        model = model.to(Args.device)

    criterion = losses.__dict__[Args.losses]()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Args.lr)
    num_train_steps = int(len(train_loader) * Args.epochs)
    num_warmup_steps = int(0.1 * Args.epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

    best_lwlrap = -np.inf
    for epoch in range(Args.start_epoch, Args.epochs):
        train_avg, train_loss = functions.train_epoch(Args, model, train_loader, criterion, optimizer, scheduler, epoch)
        valid_avg, valid_loss = functions.valid_epoch(Args, model, valid_loader, criterion, epoch)

        if Args.epoch_scheduler:
            scheduler.step()

        content = f"""
                {time.ctime()} \n
                Fold:{Args.fold}, Epoch:{epoch}, lr:{optimizer.param_groups[0]['lr']:.7}\n
                Train Loss:{train_loss:0.4f} - LWLRAP:{train_avg['lwlrap']:0.4f}\n
                Valid Loss:{valid_loss:0.4f} - LWLRAP:{valid_avg['lwlrap']:0.4f}\n
        """
        print(content)
        with open(f'{Args.save_path}/log_{Args.exp_name}.txt', 'a') as appender:
            appender.write(content+'\n')

        if valid_avg['lwlrap'] > best_lwlrap:
            print(f"########## >>>>>>>> Model Improved From {best_lwlrap} ----> {valid_avg['lwlrap']}")
            torch.save(model.state_dict(), os.path.join(Args.save_path, f'fold-{Args.fold}.bin'))
            best_lwlrap = valid_avg['lwlrap']
        #torch.save(model.state_dict(), os.path.join(args.save_path, f'fold-{args.fold}_last.bin'))


    model.load_state_dict(torch.load(os.path.join(Args.save_path, f'fold-{Args.fold}.bin'), map_location=Args.device))
    model = model.to(Args.device)

    target_cols = sub_df.columns[1:].values.tolist()
    test_pred, ids = functions.test_epoch(Args, model, test_loader)
    print(np.array(test_pred).shape)

    test_pred_df = pd.DataFrame({
        "recording_id" : sub_df.recording_id.values
    })
    test_pred = np.array(test_pred)
    for col in range(len(target_cols)):
        assert len(target_cols) == test_pred.shape[1]
        test_pred_df[target_cols[col]] = test_pred[:, col]

    test_pred_df.to_csv(os.path.join(Args.save_path, f"fold-{Args.fold}-submission.csv"), index=False)
    print(os.path.join(Args.save_path, f"fold-{Args.fold}-submission.csv"))




if __name__ == "__main__":
    for fold in range(5):
        if fold == 0:
            main(fold)
