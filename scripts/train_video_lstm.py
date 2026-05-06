# scripts/train_video_lstm.py
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from cnn_feature_extractor import load_cnn_backbone
from video_lstm_model import VideoLSTM
from video_dataset import VideoFolderDataset

def parse_args():
    p = argparse.ArgumentParser(description="Train CNN+LSTM video deepfake detector (train LSTM only)")
    p.add_argument("--data-dir", required=True, help="Folder with real/ and fake/ video subfolders")
    p.add_argument("--cnn-model", default="resnet18", help="timm model name matching your image model")
    p.add_argument("--cnn-checkpoint", required=True, help="Path to trained image model checkpoint (.pth)")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)

    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.2, help="fraction for validation split")
    p.add_argument("--out-dir", default="video_outputs", help="Where to save LSTM checkpoint and metrics")

    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=1)
    return p.parse_args()

@torch.no_grad()
def cnn_embeddings(cnn, frames_bthwc: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    frames_bthwc: [B,T,3,H,W] float 0..1
    Returns: [B,T,D]
    """
    B, T, C, H, W = frames_bthwc.shape
    x = frames_bthwc.view(B * T, C, H, W).to(device)
    emb = cnn(x)  # [B*T, D]
    D = emb.shape[-1]
    emb = emb.view(B, T, D)
    return emb

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    ds = VideoFolderDataset(args.data_dir, num_frames=args.num_frames, img_size=args.img_size)
    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Load CNN backbone (frozen)
    cnn = load_cnn_backbone(args.cnn_model, args.cnn_checkpoint, device)

    # Determine embedding dim with a single batch
    xb, yb, _ = next(iter(train_ld))
    emb = cnn_embeddings(cnn, xb, device)
    emb_dim = emb.shape[-1]

    # LSTM head
    lstm = VideoLSTM(input_dim=emb_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_classes=2).to(device)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(lstm.parameters(), lr=args.lr)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        lstm.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for xb, yb, _ in train_ld:
            yb = yb.to(device)
            with torch.no_grad():
                emb = cnn_embeddings(cnn, xb, device)  # [B,T,D]

            logits = lstm(emb)
            loss = crit(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += loss.item() * yb.size(0)
            tr_correct += (logits.argmax(dim=1) == yb).sum().item()
            tr_total += yb.size(0)

        tr_loss /= max(1, tr_total)
        tr_acc = tr_correct / max(1, tr_total)

        # Val
        lstm.eval()
        va_loss = 0.0
        va_correct = 0
        va_total = 0

        with torch.no_grad():
            for xb, yb, _ in val_ld:
                yb = yb.to(device)
                emb = cnn_embeddings(cnn, xb, device)
                logits = lstm(emb)
                loss = crit(logits, yb)

                va_loss += loss.item() * yb.size(0)
                va_correct += (logits.argmax(dim=1) == yb).sum().item()
                va_total += yb.size(0)

        va_loss /= max(1, va_total)
        va_acc = va_correct / max(1, va_total)

        row = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc}
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs} | train_acc={tr_acc:.4f} val_acc={va_acc:.4f} "
              f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")

        # Save best
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            ckpt = {
                "lstm_state": lstm.state_dict(),
                "cnn_model": args.cnn_model,
                "cnn_checkpoint": args.cnn_checkpoint,
                "num_frames": args.num_frames,
                "img_size": args.img_size,
                "emb_dim": emb_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "best_val_acc": best_val_acc,
            }
            torch.save(ckpt, out_dir / "best_lstm_video_model.pth")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved best model to: {out_dir/'best_lstm_video_model.pth'}")
    print(f"Best val acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
