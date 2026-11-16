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

    try:
        dist.init_process_group(
            backend="gloo",
            rank=process_index,
            world_size=WORLD_SIZE,
        )

        train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
        train_dataset = train_loader.dataset

        sampler = DistributedSampler(
            train_dataset,
            num_replicas=WORLD_SIZE,
            rank=process_index,
            shuffle=True,
            seed=SEED,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE // WORLD_SIZE,
            sampler=sampler,
        )

        model = DDP(CNN(in_channels=1, num_classes=10))
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

        # Only process 0 records the combined results.
        if process_index == 0:
            report = REPORT[0]

        for epoch in range(1, EPOCHS + 1):
            sampler.set_epoch(epoch)

            # Wait until all processes are ready before starting the timer.
            dist.barrier()
            epoch_start = time.perf_counter()

            total_loss, correct, count = train_one_epoch(
                model,
                train_loader,
                optimizer,
                return_totals=True,
            )

            epoch_time = time.perf_counter() - epoch_start

            # Use the slowest process's training time.
            epoch_times = torch.tensor([epoch_time], dtype=torch.float64)
            dist.all_reduce(epoch_times, op=dist.ReduceOp.MAX)
            epoch_time = epoch_times.item()

            # Add the loss sums, correct predictions, and sample counts.
            totals = torch.tensor(
                [total_loss, correct, count],
                dtype=torch.float64,
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

            total_loss = totals[0].item()
            correct = totals[1].item()
            count = totals[2].item()

            # Temporary check
            if process_index == 0:
                print(f"Training samples counted: {int(count)}")


            train_loss = total_loss / count
            train_acc = correct / count

            if process_index == 0:
                # Evaluate once using the underlying model, without DDP.
                evaluation_start = time.perf_counter()
                test_loss, test_acc = evaluate(model.module, test_loader)
                evaluation_time = time.perf_counter() - evaluation_start

                report["train_loss"].append(train_loss)
                report["train_acc"].append(train_acc)
                report["test_loss"].append(test_loss)
                report["test_acc"].append(test_acc)
                report["epoch_time"].append(epoch_time)
                report["evaluation_time"].append(evaluation_time)

                print(
                    f"epoch {epoch:2d} | "
                    f"train: loss {train_loss:.3f}, accuracy {train_acc:.3f} | "
                    f"test: loss {test_loss:.3f}, accuracy {test_acc:.3f} | "
                    f"training {epoch_time:.1f} s | "
                    f"evaluation {evaluation_time:.1f} s"
                )

            # Wait until process 0 finishes evaluating.
            dist.barrier()

        if process_index == 0:
            report["avg_epoch_time"] = sum(report["epoch_time"]) / EPOCHS
            REPORT[0] = report

    finally:
        # Close the process group when this worker exits.
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    download_dataset()

    with mp.Manager() as mg:
        REPORT = mg.list()

        # Store one combined report instead of one report per process.
        REPORT.append({
            "method": f"ddp_{WORLD_SIZE}",
            "world_size": WORLD_SIZE,
            "batch_size": BATCH_SIZE, "lr": LR, "epochs": EPOCHS,
            "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [],
            "epoch_time": [],
            "evaluation_time": [],
        })

        start = time.perf_counter()
        torch.multiprocessing.spawn(main, args=(REPORT,), nprocs=WORLD_SIZE) # spawned processes don't share our globals, so inject REPORT into each one
        total_time = time.perf_counter() - start

        report = REPORT[0]

    report["total_time"] = total_time
    report["final_acc"] = report["test_acc"][-1]
    save(OUT, filename=f"DDP_{WORLD_SIZE}", report=report)

    print(f"Final test accuracy: {report['final_acc']:.3f}")
    print(f"Avg epoch time: {report['avg_epoch_time']:.1f}s")
    print(f"Total time: {report['total_time']:.1f}s")