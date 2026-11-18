"""DCSA manual gradient synchronization: parameter server or ring all-reduce.

Run from project root, e.g.:
 python code/train_ring_ps.py --method ps --world-size 4
 python code/train_ring_ps.py --method ring --world-size 4

This example uses rank 0 as BOTH a training worker and the PS coordinator.
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils.utils import CNN, download_dataset, evaluate, get_dataloaders, save

EPOCHS = 5
BATCH_SIZE = 128
LR = 0.01
PORT = 29501
SEED = 0

def flatten_gradients(model):
    """Make one contiguous 1-D tensor of all gradients."""
    gradient_parts = []
    for parameter in model.parameters():
        gradient_parts.append(parameter.grad.reshape(-1))
    return torch.cat(gradient_parts)


def set_gradients(model, flat):
    """Copy synchronized gradients back into each parameter's .grad."""
    offset = 0
    for p in model.parameters():
        count = p.numel()
        p.grad.copy_(flat[offset:offset + count].view_as(p))
        offset += count


def parameter_server(gradients, process_rank, world_size):
    """Central rank 0 gathers, averages, and broadcasts using send/recv."""
    if process_rank == 0:
        gradient_sum = gradients.clone()
        for worker_rank in range(1, world_size):
            worker_gradients = torch.empty_like(gradients)
            dist.recv(worker_gradients, src=worker_rank)
            gradient_sum.add_(worker_gradients)
        gradients.copy_(gradient_sum / world_size)
        for worker_rank in range(1, world_size):
            dist.send(gradients, dst=worker_rank)
    else:
        dist.send(gradients, dst=0)
        dist.recv(gradients, src=0)
    return gradients


def ring_all_reduce(flat, rank, world_size):
    """Manual ring reduce-scatter followed by ring all-gather.

    Only point-to-point send/recv are used for gradient synchronization.
    """
    # tensor_split permits a final chunk that differs in length.
    chunks = [c.contiguous().clone() for c in torch.tensor_split(flat, world_size)]
    prev_rank = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size

    # Phase 1: reduce-scatter. After W-1 rounds, rank r owns reduced chunk r+1 (mod W).
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size
        incoming = torch.empty_like(chunks[recv_idx])
        ops = [
            dist.P2POp(dist.isend, chunks[send_idx], next_rank),
            dist.P2POp(dist.irecv, incoming, prev_rank),
        ]
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        chunks[recv_idx].add_(incoming)

    # Phase 2: all-gather. Circulate fully-reduced chunks to all ranks.
    for step in range(world_size - 1):
        send_idx = (rank - step + 1) % world_size
        recv_idx = (rank - step) % world_size
        incoming = torch.empty_like(chunks[recv_idx])
        ops = [
            dist.P2POp(dist.isend, chunks[send_idx], next_rank),
            dist.P2POp(dist.irecv, incoming, prev_rank),
        ]
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        chunks[recv_idx] = incoming

    flat.copy_(torch.cat(chunks) / world_size)
    return flat


def worker(rank, cfg):
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    world_size = cfg["world_size"]
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        train_loader_original, test_loader = get_dataloaders(
            batch_size=cfg["batch_size"], data_dir=cfg["data_dir"]
        )
        sampler = DistributedSampler(
            train_loader_original.dataset, num_replicas=world_size,
            rank=rank, shuffle=True, seed=cfg["seed"]
        )
        train_loader = DataLoader(
            train_loader_original.dataset,
            batch_size=cfg["batch_size"] // world_size,
            sampler=sampler, num_workers=0,
        )
        model = CNN(in_channels=1, num_classes=10)
        # Make model parameters exactly identical across ranks at initialization.
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)
        report = {
            "method": cfg["method"], "world_size": world_size,
            "batch_size": cfg["batch_size"], "lr": cfg["lr"],
            "epochs": cfg["epochs"], "train_loss": [],
            "train_acc": [], "test_loss": [], "test_acc": [],
            "epoch_time": [], "sync_time": [],
        }
        run_start = time.perf_counter()
        for epoch in range(1, cfg["epochs"] + 1):
            sampler.set_epoch(epoch)
            model.train()
            dist.barrier()
            epoch_start = time.perf_counter()
            loss_sum = 0.0
            correct = 0
            count = 0
            local_sync_seconds = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                flat = flatten_gradients(model)
                sync_start = time.perf_counter()
                if cfg["method"] == "ps":
                    flat = parameter_server(flat, rank, world_size)
                else:
                    flat = ring_all_reduce(flat, rank, world_size)
                local_sync_seconds += time.perf_counter() - sync_start
                set_gradients(model, flat)
                optimizer.step()
                bs = len(labels)
                count += bs
                loss_sum += loss.item() * bs
                correct += (outputs.argmax(dim=1) == labels).sum().item()

            local_time = time.perf_counter() - epoch_start
            # These two collectives are for reporting, not gradient sync.
            all_stats = torch.tensor([loss_sum, float(correct), float(count)], dtype=torch.float64)
            dist.all_reduce(all_stats, op=dist.ReduceOp.SUM)
            perf = torch.tensor([local_time, local_sync_seconds], dtype=torch.float64)
            dist.all_reduce(perf, op=dist.ReduceOp.MAX)
            train_loss = (all_stats[0] / all_stats[2]).item()
            train_acc = (all_stats[1] / all_stats[2]).item()
            if rank == 0:
                test_loss, test_acc = evaluate(model, test_loader)
                report["train_loss"].append(train_loss)
                report["train_acc"].append(train_acc)
                report["test_loss"].append(test_loss)
                report["test_acc"].append(test_acc)
                report["epoch_time"].append(perf[0].item())
                report["sync_time"].append(perf[1].item())
                print(f"{cfg['method']} | epoch {epoch:2d} | "
                      f"train loss {train_loss:.3f}, acc {train_acc:.3f} | "
                      f"test loss {test_loss:.3f}, acc {test_acc:.3f} | "
                      f"train {perf[0].item():.1f}s | sync ~{perf[1].item():.1f}s")
            # Keep rank 0's evaluation from overlapping with other ranks' next epoch.
            dist.barrier()

        if rank == 0:
            report["total_time"] = time.perf_counter() - run_start
            report["final_acc"] = report["test_acc"][-1]
            report["avg_epoch_time"] = sum(report["epoch_time"]) / cfg["epochs"]
            report["avg_sync_time"] = sum(report["sync_time"]) / cfg["epochs"]
            save(cfg["out_dir"], f"{cfg['method']}_{world_size}", report)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ps", "ring"], required=True)
    parser.add_argument("--world-size", type=int, choices=[2, 4], default=2)
    args = parser.parse_args()
    if BATCH_SIZE % args.world_size:
        parser.error("batch size must be divisible by world size")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(PORT)
    data_dir = str(ROOT.parent / "data")  # matches utils.py's default ../data from project root
    download_dataset(data_dir=data_dir)
    cfg = dict(method=args.method, world_size=args.world_size,
               epochs=EPOCHS, batch_size=BATCH_SIZE,
               lr=LR, seed=SEED, data_dir=data_dir,
               out_dir=str(ROOT / "results"))
    mp.spawn(worker, args=(cfg,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
