import time
import sys
from pathlib import Path
import torch
import random
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils.utils import CNN, evaluate, save, train_one_epoch, download_dataset, get_dataloaders


EPOCHS = 5
LR = 0.01
BATCH_SIZE = 128
THREADS = 1
SEED = 0
OUT = "./results/"


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    random.seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")
    print(f"Running on {device}.")

    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    model = CNN(in_channels=1, num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        epoch_time = time.time() - epoch_start
        test_loss, test_acc = evaluate(model, test_loader)

        REPORT["train_loss"].append(train_loss)
        REPORT["train_acc"].append(train_acc)
        REPORT["test_loss"].append(test_loss)
        REPORT["test_acc"].append(test_acc)
        REPORT["epoch_time"].append(epoch_time)

        print(f"epoch {epoch:2d} | train: loss {train_loss:.3f}, accuracy {train_acc:.3f} | test: loss {test_loss:.3f}, accuracy {test_acc:.3f} | {epoch_time:.1f} s")

    REPORT["avg_epoch_time"] = sum(REPORT["epoch_time"]) / EPOCHS


if __name__ == "__main__":
    download_dataset()
    REPORT = {
        "method": "single",
        "batch_size": BATCH_SIZE, "lr": LR, "epochs": EPOCHS,
        "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [],
        "epoch_time": [],
    }
    start = time.time()
    main()
    total_time = time.time() - start
    REPORT["total_time"] = total_time
    REPORT["final_acc"] = REPORT["test_acc"][-1]
    save(OUT, filename="single", report=REPORT)
    print(f"Final test accuracy: {REPORT['final_acc']:.3f}")
    print(f"Avg epoch time: {REPORT['avg_epoch_time']:.1f}s")
    print(f"Total time: {REPORT['total_time']:.1f}s")
