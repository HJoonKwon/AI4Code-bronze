import json
from pathlib import Path
from dataset import *
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from model import *
from tqdm import tqdm
import sys, os
from metrics import *
import torch
import argparse
import wandb
import config
import transformers

conf = config.Config()

WANDB_CONFIG = {
    'TRAIN_BS': conf.TRAIN_BS,
    'VALID_BS': conf.VALID_BS,
    'N_EPOCHS': conf.NB_EPOCHS,
    'ARCH': conf.MODEL_NAME,
    'MD_MAX_LEN': conf.MD_MAX_LEN,
    'TOTAL_MAX_LEN': conf.TOTAL_MAX_LEN,
    'LR': conf.LR,
    'NUM_WORKERS': conf.n_workers,
    'OPTIM': "AdamW",
    'LOSS': "L1Loss",
    'DEVICE': "cuda",
    'T_0': conf.T_0,
    'η_min': conf.η_min,
    'infra': "Kaggle",
    'competition': 'ai4code',
    '_wandb_kernel': 'joon'
}

# os.mkdir("./outputs")
# data_dir = Path('..//input/')
data_dir = conf.data_dir
wb_key = conf.wb_key
wandb.login(key=wb_key)

train_df_mark = pd.read_csv(conf.train_mark_path).drop(
    "parent_id", axis=1).dropna().reset_index(drop=True)
train_fts = json.load(open(conf.train_features_path))
val_df_mark = pd.read_csv(conf.val_mark_path).drop(
    "parent_id", axis=1).dropna().reset_index(drop=True)
val_fts = json.load(open(conf.val_features_path))
val_df = pd.read_csv(conf.val_path)

order_df = pd.read_csv("../input/train_orders.csv").set_index("id")
df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

train_ds = MarkdownDataset(train_df_mark,
                           model_name_or_path=conf.MODEL_NAME,
                           md_max_len=conf.MD_MAX_LEN,
                           total_max_len=conf.TOTAL_MAX_LEN,
                           fts=train_fts)
val_ds = MarkdownDataset(val_df_mark,
                         model_name_or_path=conf.MODEL_NAME,
                         md_max_len=conf.MD_MAX_LEN,
                         total_max_len=conf.TOTAL_MAX_LEN,
                         fts=val_fts)
train_loader = DataLoader(train_ds,
                          batch_size=conf.TRAIN_BS,
                          shuffle=True,
                          num_workers=conf.n_workers,
                          pin_memory=False,
                          drop_last=True)
val_loader = DataLoader(val_ds,
                        batch_size=conf.VALID_BS,
                        shuffle=False,
                        num_workers=conf.n_workers,
                        pin_memory=False,
                        drop_last=False)


def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


def validate(model, val_loader):
    model.eval()
    criterion = torch.nn.L1Loss()
    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []
    loss_list = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)
                loss = criterion(pred, target)
                loss_list.append(loss.detach().cpu().item())

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())
        avg_loss = np.round(np.mean(loss_list), 4)

    return np.concatenate(labels), np.concatenate(preds), avg_loss


def compute_metrics():

    pass


def yield_optimizer(model):
    """
    Returns optimizer for specific parameters
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            0.01,
        },
        {
            "params":
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay":
            0.0,
        },
    ]
    return transformers.AdamW(optimizer_parameters,
                              lr=conf.LR,
                              correct_bias=False)


def wandb_log(**kwargs):
    """
    Logs a key-value pair to W&B
    """
    for k, v in kwargs.items():
        wandb.log({k: v})


def train(model, train_loader, val_loader, epochs):
    np.random.seed(0)
    # Creating optimizer and lr schedulers

    # num_train_optimization_steps = int(args.epochs * len(train_loader) /
    #                                    args.accumulation_steps)
    num_train_optimization_steps = int(conf.NB_EPOCHS * len(train_loader) /
                                       conf.accumulation_steps)
    optimizer = yield_optimizer(model)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=conf.T_0, eta_min=conf.η_min)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.05 * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    criterion = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()

    run = wandb.init(
        project='ai4code',
        config=WANDB_CONFIG,
        group='code-bert-base',
        job_type='train',
    )

    for e in range(epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)
                loss = criterion(pred, target)
            scaler.scale(loss).backward()
            if idx % conf.accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            wandb_log(train_batch_loss=avg_loss)

            tbar.set_description(
                f"Epoch {e + 1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}"
            )

        y_val, y_pred, val_loss = validate(model, val_loader)
        val_df["pred"] = val_df.groupby(["id",
                                         "cell_type"])["rank"].rank(pct=True)
        val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred
        y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(
            list)
        score = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
        wandb_log(predidction_score=score, val_loss=val_loss)
        print("Preds score", score)
        if not os.path.isdir(conf.output_dir):
            os.mkdir(conf.output_dir)
        torch.save(model.state_dict(),
                   os.path.join(conf.output_dir, f'model_epoch{e+1}.bin'))

    return model, y_pred


model = MarkdownModel(conf.MODEL_NAME)
model = model.cuda()
model, y_pred = train(model, train_loader, val_loader, epochs=conf.NB_EPOCHS)
