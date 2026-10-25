"""Train compact 1D CNN and LSTM baselines on 250-sample beats."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from src.common import load_dataset, split_records, save_metrics, CLASSES

class ECGCNN(nn.Module):
    # Paper describes four convolutional and four fully connected layers;
    # exact dense-layer details are not fully specified, so this is a compact 1D adaptation.
    def __init__(self, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(16), nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*16, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class ECGLSTM(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(64, n_classes))
    def forward(self, x):
        x = x.transpose(1, 2)  # batch, time, channel
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def train_model(model, train_loader, val_loader, epochs, device, out_path, patience=5, min_delta=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(xb)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * len(xb)

        val_loss /= len(val_loader.dataset)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(
            f"epoch {epoch + 1:03d}/{epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}"
        )

        # Save best weights and track validation improvement
        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            torch.save(model.state_dict(), out_path)
        else:
            epochs_without_improvement += 1

        # Stop when validation loss stops improving
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best validation loss: {best_val:.4f} "
                f"at epoch {best_epoch}."
            )
            break

    # Restore best validation checkpoint
    model.load_state_dict(
        torch.load(out_path, map_location=device, weights_only=True)
    )

    print(f"Restored best model from epoch {best_epoch}.")
    return history


def predict_probs(model, loader, device):
    model.eval(); probs=[]; ys=[]
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
            ys.extend(yb.numpy())
    return np.asarray(probs), np.asarray(ys)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--out", default="results/deep")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=2020)
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    X, y, groups = load_dataset(args.data)
    train_records, val_records, test_records = split_records(groups, args.seed)
    m_train, m_val, m_test = [np.isin(groups, r) for r in (train_records, val_records, test_records)]
    class_to_id = {name: i for i, name in enumerate(CLASSES)}
    yi = np.array([class_to_id[v] for v in y], dtype=int)
    # Normalize each beat using training-set global mean/std only.
    mean = float(X[m_train].mean()); std = float(X[m_train].std() + 1e-8)
    Xn = (X - mean) / std
    def loader(mask, shuffle=False):
        xx = torch.tensor(Xn[mask, None, :], dtype=torch.float32)
        yy = torch.tensor(yi[mask], dtype=torch.long)
        return DataLoader(TensorDataset(xx, yy), batch_size=args.batch_size, shuffle=shuffle)
    train_loader = loader(m_train, True); val_loader = loader(m_val); test_loader = loader(m_test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    for name, cls in [("cnn", ECGCNN), ("lstm", ECGLSTM)]:
        print("Training", name)
        model = cls().to(device)
        model_path = out / f"{name}.pt"
        hist = train_model(model, train_loader, val_loader, args.epochs, device, model_path)
        (out / f"{name}_history.json").write_text(json.dumps(hist, indent=2))
        model.load_state_dict(torch.load(model_path, map_location=device))
        probs, truth = predict_probs(model, test_loader, device)
        pred = probs.argmax(axis=1)
        summary = save_metrics(truth, pred, out / name, prefix=name)
        np.save(out / f"{name}_test_probs.npy", probs)
        np.save(out / "test_y.npy", truth)
        print(f"{name}: accuracy={summary['accuracy']:.4f}, macro-F1={summary['macro_f1']:.4f}")
    (out / "normalization.json").write_text(json.dumps({"mean": mean, "std": std}, indent=2))
    (out / "split_records.json").write_text(json.dumps({
        "train_records": train_records.tolist(), "validation_records": val_records.tolist(),
        "test_records": test_records.tolist()
    }, indent=2))
if __name__ == "__main__":
    main()
