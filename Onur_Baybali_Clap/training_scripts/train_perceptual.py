#!/usr/bin/env python3
# coding: utf-8
# Perceptual-Only CLAP Model (based on smc_revize.py)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import time
from pprint import PrettyPrinter
import wandb
import torch
import argparse
from ruamel.yaml import YAML
from tqdm import tqdm
from loguru import logger
from data_handling.perceptual_datamodule import PerceptualDataModule
from models.audio_embedding_perceptual import ASEPerceptual
import torch.distributed as dist
from tools.optim_utils import get_optimizer, cosine_lr
from tools.utils import (
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    is_main_process,
    setup_seed,
    AverageMeter, t2a, a2t, set_logger, log_results,
)

WB_LOG = False

def train(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    
    epoch_loss = AverageMeter()
    start_time = time.time()

    if is_dist_avail_and_initialized():
        dataloader.sampler.set_epoch(epoch)

    for batch_id, (audio_feat, input_ids, idx) in tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()
        step = len(dataloader) * (epoch - 1) + batch_id
        scheduler(step)

        if is_main_process() and WB_LOG:
            wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=step)

        audio_feat = audio_feat.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        loss = model(audio_feat, input_ids, idx)
        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.cpu().item())

    elapsed_time = time.time() - start_time

    if is_main_process() and WB_LOG:
        wandb.log({"loss": epoch_loss.avg, "epoch": epoch})

    return {"loss": epoch_loss.avg, "time": elapsed_time}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="settings/perceptual.yaml", type=str)
    parser.add_argument("-n", "--exp_name", default="perceptual_only_test", type=str)
    args = parser.parse_args()

    # Load config with ruamel.yaml
    with open(args.config, "r") as f:
        yaml_loader = YAML(typ='safe', pure=True)
        config = yaml_loader.load(f)

    init_distributed_mode(config["dist_args"])
    device = torch.device(config["device"])

    seed = config["seed"] + get_rank()
    setup_seed(seed)

    if is_main_process() and WB_LOG:
        wandb.init(project="SMC", name=args.exp_name, config=config, group="Perceptual-Only")

    # Use PerceptualDataModule instead of AudioCaptionDataModule
    datamodule = PerceptualDataModule(config)
    dataloader = datamodule.train_dataloader()
    
    # Use ASEPerceptual instead of ASE
    model = ASEPerceptual(config, feature_dim=datamodule.feature_dim).to(device)

    optimizer = get_optimizer(model.parameters(),
                              lr=config["optim_args"]["lr"],
                              betas=config["optim_args"]["betas"],
                              eps=config["optim_args"]["eps"],
                              momentum=config["optim_args"]["momentum"],
                              optimizer_name=config["optim_args"]["optimizer_name"])

    scheduler = cosine_lr(optimizer,
                          base_lr=config["optim_args"]["lr"],
                          warmup_length=config["optim_args"]["warmup_epochs"] * len(dataloader),
                          steps=len(dataloader) * config["training"]["epochs"])

    start_epoch = 1
    max_epoch = config["training"]["epochs"]

    if config["resume"]:
        cp = torch.load(config.checkpoint, map_location="cpu")
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        start_epoch = cp["epoch"] + 1

    model_output_dir, log_output_dir = set_logger(args.exp_name)
    main_logger = logger.bind(indent=1)
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n' + f'{printer.pformat(config)}')
    main_logger.info(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    main_logger.info(f'Size of training set: {len(dataloader.dataset)}, size of batches: {len(dataloader)}')

    if is_dist_avail_and_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    model_without_ddp = model.module if is_dist_avail_and_initialized() else model

    if is_main_process() and WB_LOG:
        wandb.watch(model)

    val_loader = datamodule.val_dataloader()
    loss_stats, recall_stats = [], []

    for epoch in range(start_epoch, max_epoch + 1):
        main_logger.info(f'Training for epoch [{epoch}]')
        train_stats = train(model, dataloader, optimizer, scheduler, device, epoch)
        main_logger.info(f'Epoch [{epoch}] - Loss: {train_stats["loss"]:.3f}, Time: {train_stats["time"]:.1f}')

        if is_main_process():
            metrics = validate(model_without_ddp, val_loader, device)
            log_results(metrics, 'Perceptual-Only', main_logger, test=False)
            recall_stats.append(metrics["t2a"][0] + metrics["a2t"][0])
            if recall_stats[-1] >= max(recall_stats):
                torch.save({
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch
                }, f"{model_output_dir}/perceptual_only_best_model.pt")

    if is_main_process():
        main_logger.info("Evaluation start...")
        model_without_ddp.load_state_dict(torch.load(f"{model_output_dir}/perceptual_only_best_model.pt")["model"])
        metrics = validate(model_without_ddp, datamodule.test_dataloader(), device)
        log_results(metrics, 'Perceptual-Only', main_logger, test=True)
        if WB_LOG:
            wandb.finish()

@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    audio_embeds_all, text_embeds_all = [], []
    for _, (audio_feat, input_ids, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        audio_feat = audio_feat.to(device)
        input_ids = input_ids.to(device)
        
        audio_embeds = model.encode_audio(audio_feat)
        text_embeds = model.encode_text(input_ids)
        
        audio_embeds_all.append(audio_embeds.cpu())
        text_embeds_all.append(text_embeds.cpu())

    audio_embeds_all = torch.cat(audio_embeds_all).numpy()
    text_embeds_all = torch.cat(text_embeds_all).numpy()
    r1, r5, r10, r50, medr, meanr, mAP10 = t2a(audio_embeds_all, text_embeds_all)
    r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a = a2t(audio_embeds_all, text_embeds_all)
    return {"t2a": [r1, r5, r10, r50, medr, meanr, mAP10], "a2t": [r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a]}

if __name__ == '__main__':
    main()
