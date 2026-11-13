import time
import torch
import random
import numpy as np
from utils.utils import CNN, evaluate, init_dataset, save_results, train_one_epoch


EPOCHS = 5
LR = 0.01
BATCH_SIZE = 128
THREADS = 1
SEED = 0
OUT = "../results/single.json"


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    random.seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")
    print(f"Running on {device}.")

    train_loader, test_loader = init_dataset(batch_size=BATCH_SIZE)
    model = CNN(in_channels=1, num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    results = {
        "method": "single",
        "batch_size": BATCH_SIZE, "lr": LR, "epochs": EPOCHS,
        "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [],
        "epoch_time": [],
    }

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        epoch_time = time.time() - start
        test_loss, test_acc = evaluate(model, test_loader)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["epoch_time"].append(epoch_time)

        print(f"epoch {epoch:2d} | train: loss {train_loss:.3f}, accuracy {train_acc:.3f} | test: loss {test_loss:.3f}, accuracy {test_acc:.3f} | {epoch_time:.1f} s")

    results["avg_epoch_time"] = sum(results["epoch_time"]) / EPOCHS
    results["total_epochs_time"] = sum(results["epoch_time"])
    results["final_acc"] = results["test_acc"][-1]
    print(f"\nAvg epoch time: {results['avg_epoch_time']:.1f}s | Total epochs time: {results['total_epochs_time']:.1f}s | Final test accuracy: {results['final_acc']:.3f}")
    save_results(OUT, results)


if __name__ == "__main__":
    main()