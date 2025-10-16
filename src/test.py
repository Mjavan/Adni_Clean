import numpy as np
import pandas as pd
import torch
from pathlib import Path

from model import *
from data_fintune import *
from captum.attr import LRP

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import LRP
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule

import argparse

def test_func(args):
    test_loss = 0.0
    correct = 0
    total = 0

    # array of predictions
    pred_list = []
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test set
    root_dir = Path('/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability')
    ckpt_dir = root_dir / "AdniGithub"/ "adni_results" / "ckps"
    # Defining directory for saving heatmaps and lable 
    heatmap_dir = root_dir / "AdniGithub"/ "adni_results" / "heatmaps_captum"

    # We use two groups of data without corruption (False) or with corruption (True)
    train_dir = Path(root_dir) / "AdniGithub"/ "adni_results" / "split" / "train" / "False"
    data = np.load(train_dir / "train_val_splits.npz")
    X_train = data["X_train"]; y_train = data["y_train"]
    X_val   = data["X_val"];   y_val   = data["y_val"]

    # Making dataloaders
    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, batch_size=1, num_workers=4)

    # Load the best model
    # Use fine-tunned model on two groups of data without corruption (False) or with corruption (True)
    net = ResNet50Predictor(embed_dim=2048, dropout=0.5).to(device)
    model = finetune_net(net, num_classes=2).to(device)

    ckp_dir = ckpt_dir / f"model_finetun_last_2_False.pt"
    checkpoint = torch.load(ckp_dir, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print(f'weights were loaded sucssessfuly')  
    model.eval()
    prediction_list = []

    # --- prepare explainer once
    if args.expl == "ig":
        IG = IntegratedGradients(model)
    elif args.expl == "nt":
        IG = IntegratedGradients(model)
        NT = NoiseTunnel(IG)

    #att_lrp = []

    # if we want to obtain heatmaps using LRP
    #lrp = LRP(model)
    # It shows bach_idx, we wnat to visualaise heatmaps for the first batch only
    count = 0

    heat_list = []
    label_list = []
    img_list = []

    # save labels, predictions and heatmaps as numpy arrays 


    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Computing heatmaps using different explainability methods
            if count <= args.max_attr:
                # Obtain heatmaps for the first batch only
                x_attr = inputs.detach().clone().requires_grad_(True)
                targets = predicted.tolist()
                if args.expl == 'ig':
                    with torch.enable_grad():
                        print('Obtaining heatmaps using Integrated Gradients...')
                        attr = IG.attribute(x_attr, target=targets, n_steps=50)
                        heat_list.append(attr.detach().cpu().numpy())
                elif args.expl == 'nt':
                    with torch.enable_grad():
                        attr = NT.attribute(x_attr,
                                            nt_samples=8, 
                                            nt_type='smoothgrad_sq', 
                                            target=targets,
                                            baselines=torch.zeros_like(x_attr),
                                            n_steps=32,
                                            internal_batch_size=4
                                            )
                        heat_list.append(attr.detach().cpu().numpy())
                        del x_attr, attr
                    print('Heatmaps are obtained!')
            count += 1
            label_list.append(labels.cpu().numpy())
            prediction_list.append(predicted.cpu().numpy())

    
    avg_loss = test_loss / total
    accuracy = correct / total
    
    print(f'Val Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.4f}')

    # lets save labels, predictions and heatmaps as numpy arrays
    labelnp = np.concatenate(label_list)
    heatmap = np.concatenate(heat_list)
    predictions = np.concatenate(prediction_list)

    heatmap_path = heatmap_dir / f'heatmaps_{args.expl}_maxattr{args.max_attr}_last_2_False.npz'
    # ensure the folder exists
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        heatmap_path,
        heatmaps=heatmap.astype(np.float32),
        labels=labelnp.astype(np.int8),
        predictions=predictions.astype(np.int8),
        expl=str(args.expl),
        max_attr=int(args.max_attr)
    )
    

    return avg_loss, accuracy

parser = argparse.ArgumentParser(description='Obtaining heatmaps using different expl methods!')
parser.add_argument('--expl', type=str, default='nt', help='Explainability method: ig, lrp, gs, occlusion')
parser.add_argument('--max_attr', type=int, default=4, help='Maximum number of attributions to compute')

if __name__ == "__main__":
    args = parser.parse_args()
    avg_loss, acc = test_func(args)
