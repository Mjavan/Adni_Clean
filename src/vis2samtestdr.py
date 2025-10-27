import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

import os
import numpy as np
import pandas as pd
from pathlib import Path
import random
import argparse

import json
import warnings

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

from gradcam2 import *
from lrp_zennit import LRPWrapper
from model import *
from embeddingtest import *
from data import *
from utils import *


class TestStatisticBackprop:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._setup_experiment()
        self._load_checkpoint()

    def _setup_experiment(self):
        """Set random seeds, directories, and test loader."""
        self.seed = self.args.random_state
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Directory for loading checkpoints and saving outputs
        self.root_dir = Path(__file__).resolve().parents[1]
        base = os.path.join(self.root_dir, "adni_results")
        self.heatmap_dir = os.path.join(base, "heatmaps")
        self.embed_dir = os.path.join(base, "embeddings")
        self.param_dir = os.path.join(base, "params")
        self.overlay_dir = os.path.join(base, "overlay")
        self.statistic_dir = os.path.join(base, "statistics")
        self.img_dir = os.path.join(base, "images")

        os.makedirs(self.heatmap_dir, exist_ok=True)
        os.makedirs(self.embed_dir, exist_ok=True)
        os.makedirs(self.param_dir, exist_ok=True)
        os.makedirs(self.overlay_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)

        self._save_param()
        self._load_test_set()

    def _save_param(self):
        args_dict = vars(self.args)
        param_path = os.path.join(self.param_dir, f"{self.args.exp}")
        with open(f"{param_path}_params.json", "w") as f:
            json.dump(args_dict, f, indent=4)

    def _convert_to_tensor(self, group):
        # Assuming images_np is of shape (n_samples, 1, 256, 256)
        group_tensor = torch.tensor(group).unsqueeze(1).to(self.device).float()
        group_tensor = group_tensor / 255.0  # Rescale to [0, 1]
        group_tensor = group_tensor.repeat(1, 3, 1, 1)  # create image with 3 channels

        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        # Apply ImageNet normalization
        group_tensor = (group_tensor - IMAGENET_MEAN) / IMAGENET_STD

        return group_tensor

    def _load_test_set(self):
        """Load test dataset.
        m: number of samples in each group"""
        if self.args.dst == "test":
            print(f"Using test set for getting embeddings")
            root_dir = "/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability"
            test_dir = Path(root_dir) / "AdniGithub" / "adni_results" / "split" / "test" / "False" / "None"
            out_path = test_dir / "test_split.npz"
            with np.load(out_path) as f:
                print(f.files)  # -> ['test0', 'test1']
                self.group0 = f["test0"]
                self.group1 = f["test1"]
            self.group0_np = self.group0[: self.args.n]
            self.group1_np = self.group1[: self.args.m]

        elif self.args.dst == "full":
            dataset = AdniMRIDataset2D(annotations_file=self.args.annot_path, img_dir=self.args.img_path)
            self.group0, self.group1 = [], []
            for img, label in dataset:
                if label == 0:
                    self.group0.append(img)
                else:
                    self.group1.append(img)
            # concatenate list of arrays to a single numpy array
            self.group0 = np.concatenate(self.group0, axis=0)
            self.group1 = np.concatenate(self.group1, axis=0)
            # save two gropus as numpy arrays
            if self.args.sav_gr_np:
                full_path1 = os.path.join(self.img_dir, f"gr0_{len(self.group0)}.npy")
                full_path2 = os.path.join(self.img_dir, f"gr1_{len(self.group1)}.npy")
                np.save(full_path1, self.group0)
                np.save(full_path2, self.group1)
                print(f"group0 shape:{self.group0.shape}, group1 shape:{self.group1.shape}")
                print(f"group0 and group1 numpy arrays were saved!")
            # choose n,m samples from ecah group
            self.group0_np = self.group0[: self.args.n]
            self.group1_np = self.group1[: self.args.m]

        elif self.args.dst == "corr":

            if self.args.deg == "bl-test":
                print(f"Using corrupted test set for getting embeddings")
                root_dir = "/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability"
                test_dir = Path(root_dir) / "AdniGithub" / "adni_results" / "split" / "test" / "True"
                out_path = test_dir / "test_split.npz"
                with np.load(out_path) as f:
                    print(f.files)  # -> ['test0', 'test1']
                    self.group0 = f["test0"]
                    self.group1 = f["test1"]
                self.group0_np = self.group0[: self.args.n]
                self.group1_np = self.group1[: self.args.m]

            if self.args.deg == "zer-test":
                print(f"Using corrupted test set for getting embeddings")
                # root_dir = "/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability"
                root_dir = "."
                test_dir = Path(root_dir) / "adni_results" / "split" / "test" / "True"
                out_path = test_dir / "zer32" / "test_split.npz"
                print("images with patch size 32 corrupted are used")
                with np.load(out_path) as f:
                    print(f.files)  # -> ['test0', 'test1']
                    self.group0 = f["test0"]
                    self.group1 = f["test1"]

                self.group0_np = self.group0[: self.args.n]
                self.group1_np = self.group1[: self.args.m]

        # convert numpy array to tensor
        self.group0 = self._convert_to_tensor(self.group0_np)
        self.group1 = self._convert_to_tensor(self.group1_np)
        # make dataloaders for each group
        self.group0_loader = DataLoader(self.group0, batch_size=self.args.bs, shuffle=False, drop_last=True)
        self.group1_loader = DataLoader(self.group1, batch_size=self.args.bs, shuffle=False, drop_last=True)
        print(f"Data Loaders were built!")
        print(f"########################")

    def _load_checkpoint(self):
        """Load model checkpoint."""
        # This is the path for self-supervised SimCLR model
        if args.ckp == "simclr":
            print("Using self-supervised pre-trained model")
            base_path = "/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability/Retina_Codes"
            root_dir = Path(base_path)
            checkpoint_dir = root_dir / "self_supervised" / "simclr" / "simclr_ckpts"
            pre_exp = 2
            sam_dir_last = os.path.join(checkpoint_dir, f"{pre_exp}_last_sclr.pt")
            state_dict = torch.load(sam_dir_last, weights_only=False, map_location=self.device)
            # Load model
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model = SimCLR(backbone, hid_dim=2048, out_dim=128).to(self.device)
            print(f"Checkpoint loaded from {sam_dir_last}")
            model.load_state_dict(state_dict["model"])
            self.encoder = torch.nn.Sequential(*list(model.children())[:-1])
            print(f"encoder:{self.encoder}")
            print("##########################################")
            print(f"self-supervised model loaded from checkpoint-dir")
            print("##########################################")

        elif args.ckp == "fnt":
            print("Using fine-tuned model on two groups of data without corruption (False)")
            base_path = "/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability/AdniGithub"
            root_dir = Path(base_path)
            # checkpoint_dir = root_dir / 'adni_results' / 'ckps' / 'model_finetun_last_2_False.pt'
            checkpoint_dir = os.path.join(root_dir, args.model_path)
            print(f"ckp_dir:{checkpoint_dir}")
            state_dict = torch.load(checkpoint_dir, map_location=self.device)
            net = ResNet50Predictor(embed_dim=2048, dropout=0.5).to(self.device)
            model = finetune_net(net, num_classes=2).to(self.device)
            model.load_state_dict(state_dict["model_state"])
            backbone = model.feature_extractor
            self.encoder = torch.nn.Sequential(*list(backbone.children()))
            # print(f'encoder:{self.encoder}')
            print("##########################################")
            print(f"fine-tuned model loaded from checkpoint-dir")
            print("##########################################")

        elif args.ckp == "fnt_bl":
            print("Using fine-tuned model on two groups of data with corruption (True)")
            # base_path = "/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability/AdniGithub"
            base_path = "."
            root_dir = Path(base_path)
            checkpoint_dir = root_dir / "adni_results" / "ckps" / "model_finetun_last_7_True.pt"
            state_dict = torch.load(checkpoint_dir, map_location=self.device)
            net = ResNet50Predictor(embed_dim=2048, dropout=0.5).to(self.device)
            model = finetune_net(net, num_classes=2).to(self.device)
            model.load_state_dict(state_dict["model_state"])
            backbone = model.feature_extractor
            self.encoder = torch.nn.Sequential(*list(backbone.children()))
            # print(f'encoder:{self.encoder}')
            print("##########################################")
            print(f"fine-tuned model fine-tuned on blurred images loaded from checkpoint-dir")
            print("##########################################")

        elif args.ckp == "fnt_zer":
            print("Using fine-tuned model on two groups of data with corruption (True)")
            # base_path = "/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability/AdniGithub"
            base_path = "."
            root_dir = Path(base_path)
            checkpoint_dir = os.path.join(root_dir, args.model_path)
            print(f"ckp_dir:{checkpoint_dir}")
            state_dict = torch.load(checkpoint_dir, map_location=self.device)
            net = ResNet50Predictor(embed_dim=2048, dropout=0.5).to(self.device)
            model = finetune_net(net, num_classes=2).to(self.device)
            model.load_state_dict(state_dict["model_state"])
            backbone = model.feature_extractor
            self.encoder = torch.nn.Sequential(*list(backbone.children()))
            # print(f'encoder:{self.encoder}')
            print(f"fine-tuned model fine-tuned on corrupted images loaded from: {checkpoint_dir}")
            print("##########################################")

        elif args.ckp == "suppr":
            print("Using supervised pre-trained model without fine-tuning")
            base_path = "/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability/AdniGithub"
            root_dir = Path(base_path)
            checkpoint_dir = root_dir / "adni_results" / "ckps" / "resnet50_ukb_age_predict_epoch13.pth"
            weights = torch.load(checkpoint_dir, map_location=self.device)
            net = ResNet50Predictor(embed_dim=2048, dropout=0.5).to(self.device)
            net.load_state_dict(weights)
            print(f"net that was pre-trained supervised on UKB for age prediction:{net}")
            backbone = net.feature_extractor
            print(f"backbone: feature-extractor")
            self.encoder = torch.nn.Sequential(*list(backbone.children()))
            # print(f'encoder:{self.encoder}')
            print("##########################################")
            print(f"supervised-pre-trained without fine-tuning")
            print("##########################################")

    def get_mean_embeddings(self, explainer, dataloader, latent_dim=2048):
        """Takes dataloader of each group, extract embedding vectors,
        return mean embeddings"""
        # Initialize accumulators for healthy and unhealthy groups
        sum_f = torch.zeros(latent_dim, device=self.device)
        count_f = 0  # Count of samples in each group

        # Use no_grad since we don't need gradients for mean computation
        with torch.no_grad():
            for images in dataloader:
                # check if images are in 3 channels
                # images: [64, 3, 256, 256]
                images = images.to(self.device)
                print(f"images shape in get_mean_embeddings:{images.shape}")
                embeddings = explainer.forward(images)
                embeddings = embeddings.view(embeddings.size()[0], -1)
                sum_f += embeddings.sum(dim=0)  # Sum of embeddings for this batch
                count_f += embeddings.size(0)

                # Free memory immediately
                del embeddings, images

        mean_embed = sum_f / count_f if count_f > 0 else torch.zeros_like(sum_f)
        del sum_f
        torch.cuda.empty_cache()
        return mean_embed

    def backprobagate_statistics(self):
        """Calculate the test statistic for two groups of ADNI."""
        # Create explainer based on method
        if self.args.expl == "cam":
            # print(f'{self.args.expl} method was called with encoder:{self.encoder}')
            explainer = GradCAM(self.encoder, target_layer=self.args.target_layer, relu=True, device=self.device)
        elif self.args.expl == "cam++":
            print(f"cam++ method was called for visualisation.")
            print(f"###########################################")
            explainer = GradCAMPlusPlus(
                self.encoder, target_layer=self.args.target_layer, relu=True, device=self.device
            )
        elif self.args.expl == "lcam":
            print(f"LayerCam method was called for visualisation.")
            print(f"###########################################")
            explainer = LayerCAM(self.encoder, target_layer=self.args.target_layer, relu=True, device=self.device)
        elif self.args.expl == "lrp":
            print(f"LRP method was called for visualisation using zennit library.")
            print(f"###########################################")
            explainer = LRPWrapper(
                self.encoder,
                target_layer=self.args.target_layer,
                relu=True,
                device=self.device,
                composite_type=self.args.lrp_composite,
                lrp_epsilon=self.args.lrp_epsilon,
                lrp_gamma=self.args.lrp_gamma,
                input_low=self.args.lrp_input_low,
                input_high=self.args.lrp_input_high,
            )
        else:
            raise ValueError(f"Unknown explanation method: {self.args.expl}")

        # Calculate mean embeddings
        group0_mean = self.get_mean_embeddings(explainer, self.group0_loader)
        group1_mean = self.get_mean_embeddings(explainer, self.group1_loader)
        D = group0_mean - group1_mean
        print(f"group0_mean:{group0_mean.shape}")
        print(f"group1_mean:{group1_mean.shape}")
        print(f"##########################")
        test_statistic = torch.norm(D, p=2) ** 2
        print(f"test_statistic:{test_statistic:.4f}")
        print(f"##########################")
        return (test_statistic, D, explainer)

    def process_attributions(self, dataloader, explainer, D, group_id, use_squared=True):
        """
        Process and return attributions with proper MMD gradients.
        Works with both GradCAM and LRP methods.

        Args:
            dataloader: DataLoader for the group
            explainer: GradCAM or LRP explainer object
            D: Difference vector (group0_mean - group1_mean)
            group_id: 0 for group0, 1 for group1
            use_squared: If True, uses ||D||² (matching embeddingtest.py), else ||D||
        """
        attributions_list = []
        embed_list = []

        n_samples = len(dataloader.dataset)

        # Pre-compute gradient constant (stays the same for all batches)
        sign = 1.0 if group_id == 0 else -1.0
        if use_squared:
            # ∂(||D||²)/∂embedding = ±(2/n) * D
            grad_base = (2.0 / n_samples) * sign * D
            scaling = 2.0 / n_samples
        else:
            # ∂(||D||)/∂embedding = ±D/(n*||D||)
            D_norm = torch.norm(D, p=2)
            grad_base = (sign / n_samples) * (D / D_norm)
            scaling = 1.0 / n_samples

        is_lrp = isinstance(explainer, LRPWrapper)

        # Compute attribution maps for each group
        for images in dataloader:
            images = images.to(self.device)

            if is_lrp:
                # LRP approach: directly compute relevance for the projection
                attributions = explainer.compute_attributions_for_batch(
                    images, direction_vector=D, sign=sign, scaling=scaling
                )

                # Get embeddings separately for saving
                with torch.no_grad():
                    embeddings = explainer.forward(images)
                    batch_embed = embeddings.view(embeddings.size()[0], -1)
                    embed_list.append(batch_embed.detach().cpu().numpy())

                # Generate final heatmap
                attributions = explainer.generate()
                attributions_np = attributions.squeeze().cpu().detach().numpy()
                attributions_list.append(attributions_np)

            else:
                # GradCAM approach: gradient-based backpropagation
                # Forward pass
                embeddings = explainer.forward(images)
                batch_embed = embeddings.view(embeddings.size()[0], -1)

                # Save embeddings as numpy immediately and free GPU memory
                embed_list.append(batch_embed.detach().cpu().numpy())

                # Expand gradient to match batch size
                grad_per_sample = grad_base.unsqueeze(0).expand(batch_embed.size(0), -1)

                # Backward pass for this batch only
                explainer.model.zero_grad()
                batch_embed.backward(gradient=grad_per_sample, retain_graph=False)

                # Generate attribution with sign flip for group 1
                attributions = explainer.generate(flip_sign=(group_id == 1))
                attributions_np = attributions.squeeze().cpu().detach().numpy()
                attributions_list.append(attributions_np)

            # Free GPU memory immediately
            del images, attributions
            if not is_lrp:
                del embeddings, batch_embed, grad_per_sample
            torch.cuda.empty_cache()

        return np.vstack(attributions_list), np.vstack(embed_list)

    def run(self, backprop_type="test_statistic", latent_dim_idx=None, use_squared=True):
        """
        Main experiment function.

        Args:
            backprop_type: Type of backpropagation (kept for compatibility)
            latent_dim_idx: Latent dimension index (kept for compatibility)
            use_squared: If True, uses ||D||² matching embeddingtest.py, else ||D||
        """
        test_statistic, D, explainer = self.backprobagate_statistics()

        # Process attributions with proper per-batch gradients
        group0_attr, group0_embed = self.process_attributions(
            self.group0_loader, explainer, D=D, group_id=0, use_squared=use_squared
        )
        group1_attr, group1_embed = self.process_attributions(
            self.group1_loader, explainer, D=D, group_id=1, use_squared=use_squared
        )

        # compute test-statistic and p-value
        mmd = MMDTest(features_X=group0_embed, features_Y=group1_embed, n_perm=1000)
        test_statistic = mmd._compute_mmd(group0_embed, group1_embed)
        p_value = mmd._compute_p_value()
        print(f"Test statistic (MMD): {test_statistic:.4f}, p-value: {p_value:.4f}")

        # save_attributions(group0_attr, group1_attr,latent_dim_idx)
        self.m1 = group0_attr.shape[0]
        self.m2 = group1_attr.shape[0]

        # save_attributions(group0_attr, group1_attr,latent_dim_idx)
        full_path1 = os.path.join(
            self.heatmap_dir, f"gr1_{len(self.group0)}_{self.m1}_{self.args.expl}_{self.args.exp}.npy"
        )
        full_path2 = os.path.join(
            self.heatmap_dir, f"gr2_{len(self.group1)}_{self.m2}_{self.args.expl}_{self.args.exp}.npy"
        )

        np.save(full_path1, group0_attr)
        np.save(full_path2, group1_attr)

        print(f"gr1:{group0_attr.shape}")
        print(f"gr2:{group1_attr.shape}")

        print("Heatmaps were created")

        return (group0_attr, group1_attr)

    def overlay_hetmap(self, idx, alpha=0.5):
        """Overlay heatmap on the original image."""
        # Load original image
        if idx < len(self.group0):
            img0 = self.group0_np[idx]
        if idx < len(self.group1):
            img1 = self.group1_np[idx]

        # Load heatmap
        full_path1 = os.path.join(
            self.heatmap_dir, f"gr1_{len(self.group0)}_{self.m1}_{self.args.expl}_{self.args.exp}.npy"
        )
        full_path2 = os.path.join(
            self.heatmap_dir, f"gr2_{len(self.group1)}_{self.m2}_{self.args.expl}_{self.args.exp}.npy"
        )
        group0_attr = np.load(full_path1)
        group1_attr = np.load(full_path2)
        # Select corresponding heatmaps
        gcam0 = group0_attr[idx]
        gcam1 = group1_attr[idx]
        # Overlay heatmap
        _, overlaid_img0 = save_cam_with_alpha(img0, gcam0, alpha=alpha)
        _, overlaid_img1 = save_cam_with_alpha(img1, gcam1, alpha=alpha)

        # Save overlay images
        full_path1_ov = os.path.join(
            self.overlay_dir, f"gr1_{len(self.group0)}_{self.m1}_{self.args.expl}_{self.args.idx}_{self.args.exp}.png"
        )
        full_path2_ov = os.path.join(
            self.overlay_dir, f"gr2_{len(self.group1)}_{self.m2}_{self.args.expl}_{self.args.idx}_{self.args.exp}.png"
        )
        Image.fromarray(overlaid_img0).save(full_path1_ov)
        Image.fromarray(overlaid_img1).save(full_path2_ov)
        return overlaid_img0, overlaid_img1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Statistic Backpropagation")
    parser.add_argument("--exp", type=str, default="cam-fnt10-uncor", help="Experiment name")
    parser.add_argument(
        "--annot_path",
        type=str,
        default="/sc/projects/sci-lippert/chair/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256/group_by_hippocampus/adni_T1_3T_linear_annotation.csv",
        help="Path to annotations CSV file",
    )
    parser.add_argument("--sav_gr_np", type=bool, default=False, help="If we save two groups as numpy arrays")
    parser.add_argument("--corrupted", type=str, default=False, help="Use corrupted images for group 1")
    parser.add_argument("--deg", type=str, default=None, help="Degree of corruption: 4 or 8, test-4, zer-test ")
    parser.add_argument(
        "--ckp",
        type=str,
        default="fnt",
        choices=("random", "simclr", "fnt", "fnt_zer", "suppr", "fnt_bl"),
        help="If we use random model or checkpoints",
    )
    parser.add_argument(
        "--expl", type=str, default="cam", help="Explainability method", choices=["cam", "ig", "cam++", "lcam", "lrp"]
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default="/sc/projects/sci-lippert/chair/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256/group_by_hippocampus",
        help="Path to image directory",
    )
    parser.add_argument("--n", type=int, default=100, help="Number of samples in group 0")
    parser.add_argument("--m", type=int, default=100, help="Number of samples in group 1")
    parser.add_argument("--bs", type=int, default=100, help="Batch size for DataLoader")
    parser.add_argument(
        "--dst",
        type=str,
        default="test",
        choices=("full", "test", "corr"),
        help="Test set that we want to use for getting embeddings",
    )
    parser.add_argument("--idx", type=int, default=0, help="Index of the image for overlaying")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--model_path",
        type=str,
        default="adni_results/ckps/model_finetun_last_10_False.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--target_layer",
        type=str,
        default="0.7.2.conv3",
        choices=("0.7.2.conv3", "7.2.conv3"),
        help="Target layer for GradCAM, if suppre: 7.2.conv3",
    )

    # LRP-specific arguments
    parser.add_argument(
        "--lrp_composite",
        type=str,
        default="epsilon_plus_flat",
        choices=["epsilon_plus_flat", "epsilon_gamma_box", "epsilon_alpha2beta1"],
        help="LRP composite type for rule selection",
    )
    parser.add_argument(
        "--lrp_epsilon",
        type=float,
        default=1e-6,
        help="Epsilon value for LRP numerical stability",
    )
    parser.add_argument(
        "--lrp_gamma",
        type=float,
        default=0.25,
        help="Gamma value for LRP gamma rule",
    )
    parser.add_argument(
        "--lrp_input_low",
        type=float,
        default=-0.0,
        help="Lower bound for input normalization in ZBox rule",
    )
    parser.add_argument(
        "--lrp_input_high",
        type=float,
        default=1.0,
        help="Upper bound for input normalization in ZBox rule",
    )

    args = parser.parse_args()

    experiment = TestStatisticBackprop(args)

    # Run experiment
    group0_attr, group1_attr = experiment.run()

    # Overlay heatmap on original image
    ov1, ov2 = experiment.overlay_hetmap(idx=args.idx, alpha=0.5)
