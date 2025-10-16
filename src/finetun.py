import torch
import numpy as np
import pandas as pd
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path
import os

from model import *
from data_fintune import *
import argparse

def save_checkpoint(path, model, optimizer, epoch, best_val_loss, scheduler=None, extra=None):
    # If using DataParallel/DistributedDataParallel, save the underlying module
    payload = {
        "epoch": epoch,                       # 1-based epoch number
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val_loss,
    }
    if extra: payload.update(extra)
    torch.save(payload, path)


def finetune(num_epochs=10, batch_size=32, args=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Loading data 
    root_dir = Path('/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability')
    ckpt_dir = root_dir / "AdniGithub"/ "adni_results" / "ckps"
    if args.corrupted:
        print(f'upload corrupted images {args.deg}')
        train_dir = Path(root_dir) / "AdniGithub"/ "adni_results" / "split" / "train" / str(args.corrupted) / str(args.deg)
    else:
        print('upload non-corrupted images')
        train_dir = Path(root_dir) / "AdniGithub"/ "adni_results" / "split" / "train" / str(args.corrupted)/ str(args.deg)
    loss_dir = root_dir / "AdniGithub"/ "adni_results" / "loss"
    data = np.load(train_dir / "train_val_splits.npz")
    X_train = data["X_train"]; y_train = data["y_train"]
    X_val   = data["X_val"];   y_val   = data["y_val"]

    # Making dataloaders
    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, batch_size=32, num_workers=4)


    # path where I saved pre-trained model on Diabetic Rethinopathy dataset => self-supervised SimCLR
    if args.pre=='selfsup':
        print('Using self-supervised pre-trained model')
        base_path = "/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability/Retina_Codes"
        root_dir = Path(base_path)
        checkpoint_dir = root_dir / 'self_supervised' / 'simclr' / 'simclr_ckpts'
        pre_exp = 2
        sam_dir_last = os.path.join(checkpoint_dir, f'{pre_exp}_last_sclr.pt')
        state_dict = torch.load(sam_dir_last, weights_only=False, map_location=device) 
        # pre-trained resnet50 backbone which is used in SimCLR
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        net = SimCLR(backbone, hid_dim=2048, out_dim=128).to(device)
        net.load_state_dict(state_dict)
        # Best case: networksimclr exposes .encoder (pre-projection)
        # Encoder backbone: we remove projecttion head from SimCLR
        encoder = net.encoder

    elif args.pre=='sup':
        print('Using supervised pre-trained model')
        base_path = Path('../adni_results/ckps')
        checkpoint_dir = base_path / 'resnet50_ukb_age_predict_epoch13.pth'
        net = ResNet50Predictor(embed_dim=2048, dropout=0.5).to(device)
        weights = torch.load(checkpoint_dir, map_location=device)
        net.load_state_dict(weights)
        encoder = net
        print('supervised model loaded')

    # We freeze all layers except the final layer (linear probing)
    def freeze_all(m): 
        for p in m.parameters(): p.requires_grad = False

    # Unfreeze the last layers of the model if needed
    def unfreeze_module(m):
        for p in m.parameters(): p.requires_grad = True

    # Option 1: Linear probe only (recommended start)
    if args.freez_all:
        freeze_all(encoder)
        print('All layers were freezed! except the linear layer on top of that!')
    else:
        print('All layers are trainable!')
        # Option 2: Unfreeze all layers (after probe baseline)
        unfreeze_module(encoder)
        print('All layers were unfreezed!')
    model = finetune_net(encoder, num_classes=2).to(device)
    # Only model.linear has trainable params
    # Option 2: Unfreeze last block as needed (after probe baseline)
    # e.g., for ResNet-style encoders:
    # unfreeze_module(encoder.layer4)  # keep earlier layers frozen

    # Linear probe only:
    opt = torch.optim.AdamW(model.linear.parameters(), lr=1e-3, weight_decay=1e-4)

    # Defining loss function
    # Compute the cross-entropy loss between the predicted logits and the true labels
    loss_fn = torch.nn.CrossEntropyLoss()
    best_val = float('inf'); best_state=None

    loss_tr_list = []
    loss_val_list = []

    for epoch in range(num_epochs):
        model.train() 
        loss_epoch = 0
        correct_tr = 0
        n_train = 0
        # model.encoder.eval()
        for xb, yb in train_loader:
            xb, yb = xb.to(device).float(), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            # accumulate loss
            bs = xb.size(0)
            loss_epoch += loss.item()* bs 
            n_train += bs
            pred_tr = logits.argmax(1); correct_tr += (pred_tr==yb).sum().item()
            opt.zero_grad(); loss.backward(); opt.step()
        loss_epoch /= n_train; train_acc = correct_tr / n_train
        loss_tr_list.append(loss_epoch)
        # validate
        model.eval(); val_loss=0; correct=0; n=0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device).float(), yb.to(device)
                logits = model(xb)
                val_loss += loss_fn(logits, yb).item()*xb.size(0)
                pred = logits.argmax(1); correct += (pred==yb).sum().item(); n+=xb.size(0)
        val_loss/=n; val_acc=correct/n
        loss_val_list.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
            ckpt_dir / f"model_finetun_best_{args.exp}_{args.corrupted}.pt",
            model=model,
            optimizer=opt,
            scheduler=None,  # or pass your scheduler if you have one
            epoch=epoch + 1,
            best_val_loss=best_val)

        # print train and val losses and val accuracy for each batch
        print(f'Epoch {epoch+1}/{num_epochs} | Train loss: {loss_epoch:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} | Train acc: {train_acc:.4f}')
        # save last model 
        # -------- always save "last" for this epoch --------
        save_checkpoint(
        ckpt_dir / f"model_finetun_last_{args.exp}_{args.corrupted}.pt",
        model=model,
        optimizer=opt,
        scheduler=None,  # or pass your scheduler if you have one
        epoch=epoch + 1,
        best_val_loss=best_val,
    )
        
    # save losses
    stem = loss_dir / f"loss_{args.exp}_{args.corrupted}"
    np.savez_compressed(
    stem.with_suffix(".npz"),
    loss_tr=np.asarray(loss_tr_list, dtype=float),
    loss_val=np.asarray(loss_val_list, dtype=float),)


    # load best
    # model.load_state_dict({k:v.to(device) for k,v in best_state.items()})

# here we can load test set and evaluate model on it
parser = argparse.ArgumentParser(description='Fine tuning pre-trained model on datasets!')
parser.add_argument('--exp', type=int, default= 10, help='Experiment number for fine-tuning')
parser.add_argument('--pre', type=str, default= 'sup', help='Type of pre-trained model: sup or selsup')
parser.add_argument('--corrupted', type=str, default=False, help='Use corrupted images for group 1')
parser.add_argument('--freez_all', type=bool, default=True, help='If we want to freeze all layers except the linear layer on top')
parser.add_argument('--deg', type=str, default=None, help='Degree of corruption: 4 or 8 or None, if we do not use corrupted images')      

if __name__=="__main__":
    
    args = parser.parse_args()
    finetune(num_epochs=50, batch_size=32, args=args)







    
    



    

