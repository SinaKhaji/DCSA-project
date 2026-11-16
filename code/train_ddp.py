import os
import time
import torch
import random
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.utils import CNN, evaluate, save, train_one_epoch, download_dataset, get_dataloaders


WORLD_SIZE = 2          # number of processes: run with 2, then 4
EPOCHS = 5
LR = 0.01
BATCH_SIZE = 128
THREADS = 1
SEED = 0
OUT = "../results/"

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"


def main(process_index, REPORT):
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    random.seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")
    print(f"Running on {device}.")

    dist.init_process_group(backend="gloo", rank=process_index, world_size=WORLD_SIZE)

    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    train_dataset = train_loader.dataset
    sampler = DistributedSampler(train_dataset, num_replicas=WORLD_SIZE, rank=process_index, shuffle=True, seed=SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE // WORLD_SIZE, sampler=sampler)
    model = DDP(CNN(in_channels=1, num_classes=10))
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    report = REPORT[process_index]

    for epoch in range(1, EPOCHS + 1):
        sampler.set_epoch(epoch) #use a new distributed shuffle each epoch
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        epoch_time = time.time() - epoch_start
        test_loss, test_acc = evaluate(model, test_loader)

        report["train_loss"].append(train_loss)
        report["train_acc"].append(train_acc)
        report["test_loss"].append(test_loss)
        report["test_acc"].append(test_acc)
        report["epoch_time"].append(epoch_time)

        print(f"epoch {epoch:2d} | train: loss {train_loss:.3f}, accuracy {train_acc:.3f} | test: loss {test_loss:.3f}, accuracy {test_acc:.3f} | {epoch_time:.1f} s")

    report["avg_epoch_time"] = sum(report["epoch_time"]) / EPOCHS
    REPORT[process_index] = report


if __name__ == "__main__":
    download_dataset()
    mg = mp.Manager()
    REPORT = mg.list()
    for i in range(WORLD_SIZE):
        REPORT.append({
            "method": f"ddp_{WORLD_SIZE}", "index":i,
            "batch_size": BATCH_SIZE, "lr": LR, "epochs": EPOCHS,
            "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [],
            "epoch_time": [],
        })
    start = time.time()
    torch.multiprocessing.spawn(main, args=(REPORT,), nprocs=WORLD_SIZE) # spawned processes don't share our globals, so inject REPORT into each one
    total_time = time.time() - start
    for i in range(WORLD_SIZE):
        report = REPORT[i]
        report["total_time"] = total_time
        report["final_acc"] = report["test_acc"][-1]
        save(OUT, filename=f"DDP_{WORLD_SIZE}_{i}", report=report)
    mg.shutdown()
    print(f"Final test accuracy: {report['final_acc']:.3f}")
    print(f"Avg epoch time: {report['avg_epoch_time']:.1f}s")
    print(f"Total time: {report['total_time']:.1f}s")