"""
Explanation methods for the fine-tuned classifier using Captum attribution techniques.

This script loads a fine-tuned ResNet50 binary classifier and generates attributions
for individual images using various explanation methods (LRP, Guided Backprop, Input×Gradient),
treating it as a standard classification task (not test statistic).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from collections import OrderedDict

import os
import numpy as np
from pathlib import Path
import random
import argparse
import json
import warnings

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

from captum.attr import LRP, InputXGradient, GuidedBackprop
from captum.attr._core.lrp import EpsilonRule
from model import ResNet50Predictor, finetune_net
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class SupervisedExplainer:
    """Explanation generator for supervised binary classifier using various attribution methods."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._setup_experiment()
        self._load_checkpoint()
        self._initialize_explainer()

    def _setup_experiment(self):
        """Set random seeds and directories."""
        self.seed = self.args.random_state
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Directory setup
        self.root_dir = Path(__file__).resolve().parents[1]
        base = os.path.join(self.root_dir, "adni_results")
        self.heatmap_dir = os.path.join(base, "heatmaps_classifier")
        self.param_dir = os.path.join(base, "params")
        self.overlay_dir = os.path.join(base, "overlay_classifier")

        os.makedirs(self.heatmap_dir, exist_ok=True)
        os.makedirs(self.param_dir, exist_ok=True)
        os.makedirs(self.overlay_dir, exist_ok=True)

        self._save_param()
        self._load_test_set()

    def _save_param(self):
        """Save experiment parameters."""
        args_dict = vars(self.args)
        param_path = os.path.join(self.param_dir, f"{self.args.explanation_method}")
        with open(f"{param_path}_params.json", "w") as f:
            json.dump(args_dict, f, indent=4)

    def _convert_to_tensor(self, group):
        """Convert numpy array to normalized tensor."""
        group_tensor = torch.tensor(group).unsqueeze(1).to(self.device).float()
        group_tensor = group_tensor / 255.0
        group_tensor = group_tensor.repeat(1, 3, 1, 1)

        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        group_tensor = (group_tensor - IMAGENET_MEAN) / IMAGENET_STD

        return group_tensor

    def _load_test_set(self):
        """Load test dataset."""
        if self.args.dst == "corr" and self.args.deg == "zer-test":
            root_dir = "."
            test_dir = Path(root_dir) / "adni_results" / "split" / "test" / "True"
            out_path = test_dir / "zer32" / "test_split.npz"

            with np.load(out_path) as f:
                self.group0 = f["test0"]
                self.group1 = f["test1"]

            self.group0_np = self.group0[: self.args.n]
            self.group1_np = self.group1[: self.args.m]
        else:
            raise ValueError(f"Only dst='corr' and deg='zer-test' are supported")

        print(f"Loaded {len(self.group0_np)} samples from group 0")
        print(f"Loaded {len(self.group1_np)} samples from group 1")

    def _load_checkpoint(self):
        """Load model checkpoint."""
        print("Loading fine-tuned model")
        base_path = "."
        root_dir = Path(base_path)
        checkpoint_dir = os.path.join(root_dir, self.args.model_path)
        print(f"ckp_dir: {checkpoint_dir}")

        class LightResNetClassifier(nn.Module):
            def __init__(self, feature_extractor, num_classes=2):
                super().__init__()
                self.feature_extractor = feature_extractor
                self.fc = nn.Linear(2048, num_classes)

            def forward(self, x):
                x = self.feature_extractor(x)
                x = x.view(x.size(0), -1)  # replaces nn.Flatten()
                return self.fc(x)

        # Load the saved state dict from the file
        state_dict = torch.load(checkpoint_dir, map_location=self.device, weights_only=False)
        old_model_state = state_dict["model_state"]

        # Create a new, empty state_dict to hold the remapped keys
        new_state_dict = OrderedDict()

        # Iterate over the old state_dict and copy/remap keys
        for k, v in old_model_state.items():
            if k.startswith("model.feature_extractor."):
                # Remap: "model.feature_extractor.X" -> "feature_extractor.X"
                new_key = k.replace("model.feature_extractor.", "feature_extractor.", 1)
                new_state_dict[new_key] = v
            elif k.startswith("linear."):
                # Remap: "linear.X" -> "fc.X"
                new_key = k.replace("linear.", "fc.", 1)
                new_state_dict[new_key] = v
            # Ignore all other keys (like "model.predictor.*")

        base = ResNet50Predictor(embed_dim=2048).feature_extractor  # no predictor
        self.model = LightResNetClassifier(base, num_classes=2).to(self.device)

        # Load the *new, remapped* state_dict
        # Use strict=True to ensure all keys match perfectly after remapping
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=True)

        print("State dict remapped and loaded.")
        print("Missing keys (should be empty):", missing)
        print("Unexpected keys (should be empty):", unexpected)

        self.model.eval()

    def _initialize_explainer(self):
        """Initialize explanation method based on args.explanation_method."""
        method = self.args.explanation_method.lower()

        if method == "lrp":
            self.explainer = LRP(self.model)
            print(f"Initialized LRP explainer")
        elif method == "guided_backprop":
            self.explainer = GuidedBackprop(self.model)
            print(f"Initialized Guided Backpropagation explainer")
        elif method == "input_x_gradient":
            self.explainer = InputXGradient(self.model)
            print(f"Initialized Input × Gradient explainer")
        else:
            raise ValueError(
                f"Unknown explanation method: {self.args.explanation_method}. "
                f"Supported methods: 'lrp', 'guided_backprop', 'input_x_gradient'"
            )

    def compute_attributions(self, images, target_class):
        """
        Compute attributions for a batch of images using the selected explanation method.

        Args:
            images: Tensor of images (B, C, H, W)
            target_class: Target class for attribution (0 or 1)

        Returns:
            attributions: Attribution maps (B, C, H, W)
            logits: Model predictions (B, num_classes)
        """
        # Set requires_grad for input
        images = images.requires_grad_(True)

        # Get model predictions
        with torch.no_grad():
            logits = self.model(images)

        # Compute attributions for the target class using the initialized explainer
        attributions = self.explainer.attribute(images, target=target_class)

        return attributions, logits

    def process_group(self, group_np, group_label, group_name):
        """
        Process a group of images and compute attributions using the selected explanation method.

        Args:
            group_np: Numpy array of images
            group_label: Class label for this group (0 or 1)
            group_name: Name for saving ("group0" or "group1")

        Returns:
            attributions_np: Numpy array of attributions
            logits_np: Numpy array of predictions
        """
        # Convert to tensor
        images = self._convert_to_tensor(group_np)

        # Process in batches
        batch_size = self.args.bs
        n_samples = len(images)
        n_batches = (n_samples + batch_size - 1) // batch_size

        all_attributions = []
        all_logits = []

        print(f"Processing {group_name} with {n_samples} samples in {n_batches} batches")

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_images = images[start_idx:end_idx]

            # Compute attributions
            attributions, logits = self.compute_attributions(batch_images, group_label)

            # Store results
            all_attributions.append(attributions.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())

            # Clear GPU memory
            del attributions, logits, batch_images
            torch.cuda.empty_cache()

            if (i + 1) % 10 == 0:
                print(f"  Processed batch {i+1}/{n_batches}")

        # Concatenate all batches
        attributions_np = np.vstack(all_attributions)
        logits_np = np.vstack(all_logits)

        print(f"{group_name} processing complete")
        print(f"  Attributions shape: {attributions_np.shape}")
        print(f"  Logits shape: {logits_np.shape}")

        return attributions_np, logits_np

    def generate_heatmap(self, attributions):
        """
        Generate heatmap from attributions.

        Args:
            attributions: Attributions (C, H, W) or (1, C, H, W)

        Returns:
            heatmap: 2D heatmap (H, W)
        """
        # If batch dimension exists, squeeze it
        if attributions.ndim == 4:
            attributions = attributions.squeeze(0)

        # Sum across color channels and take absolute value
        heatmap = np.abs(attributions).sum(axis=0)

        # Normalize to [0, 1]
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        if heatmap_max - heatmap_min > 1e-8:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

        return heatmap

    def overlay_heatmap(self, image, heatmap, alpha=0.5):
        """
        Overlay heatmap on original image.

        Args:
            image: Original image (H, W) in range [0, 255]
            heatmap: Attribution heatmap (H, W) in range [0, 1]
            alpha: Transparency of heatmap overlay

        Returns:
            overlaid: RGB image with heatmap overlay
        """
        # Normalize image to [0, 1]
        image_norm = image.astype(np.float32) / 255.0

        # Apply colormap to heatmap
        colormap = cm.get_cmap("jet")
        heatmap_colored = colormap(heatmap)[:, :, :3]  # Remove alpha channel

        # Blend image and heatmap
        overlaid = (1 - alpha) * np.stack([image_norm] * 3, axis=-1) + alpha * heatmap_colored
        overlaid = (overlaid * 255).astype(np.uint8)

        return overlaid

    def run(self):
        """Main execution function."""
        print("=" * 50)
        print(f"Running Supervised Classifier Explanation ({self.args.explanation_method})")
        print("=" * 50)

        # Process group 0 (label 0)
        group0_attr, group0_logits = self.process_group(self.group0_np, 0, "group0")

        # Process group 1 (label 1)
        group1_attr, group1_logits = self.process_group(self.group1_np, 1, "group1")

        # Save raw attributions
        attr_path0 = os.path.join(self.heatmap_dir, f"group0_{self.args.explanation_method}_raw_attr.npy")
        attr_path1 = os.path.join(self.heatmap_dir, f"group1_{self.args.explanation_method}_raw_attr.npy")
        np.save(attr_path0, group0_attr)
        np.save(attr_path1, group1_attr)

        # Save logits
        logits_path0 = os.path.join(self.heatmap_dir, f"group0_{self.args.explanation_method}_logits.npy")
        logits_path1 = os.path.join(self.heatmap_dir, f"group1_{self.args.explanation_method}_logits.npy")
        np.save(logits_path0, group0_logits)
        np.save(logits_path1, group1_logits)

        # Generate and save heatmaps
        print("\nGenerating heatmaps...")
        group0_heatmaps = np.array([self.generate_heatmap(attr) for attr in group0_attr])
        group1_heatmaps = np.array([self.generate_heatmap(attr) for attr in group1_attr])

        heatmap_path0 = os.path.join(self.heatmap_dir, f"group0_{self.args.explanation_method}_heatmaps.npy")
        heatmap_path1 = os.path.join(self.heatmap_dir, f"group1_{self.args.explanation_method}_heatmaps.npy")
        np.save(heatmap_path0, group0_heatmaps)
        np.save(heatmap_path1, group1_heatmaps)

        print(f"\nSaved attributions and heatmaps:")
        print(f"  Group 0: {attr_path0}")
        print(f"  Group 1: {attr_path1}")

        # Print prediction statistics
        print("\n" + "=" * 50)
        print("Prediction Statistics")
        print("=" * 50)
        probs0 = torch.softmax(torch.from_numpy(group0_logits), dim=1).numpy()
        probs1 = torch.softmax(torch.from_numpy(group1_logits), dim=1).numpy()

        print(f"Group 0 (true label=0):")
        print(f"  Mean probability for class 0: {probs0[:, 0].mean():.4f}")
        print(f"  Mean probability for class 1: {probs0[:, 1].mean():.4f}")
        print(f"  Accuracy: {(probs0.argmax(axis=1) == 0).mean():.4f}")

        print(f"\nGroup 1 (true label=1):")
        print(f"  Mean probability for class 0: {probs1[:, 0].mean():.4f}")
        print(f"  Mean probability for class 1: {probs1[:, 1].mean():.4f}")
        print(f"  Accuracy: {(probs1.argmax(axis=1) == 1).mean():.4f}")

        return group0_heatmaps, group1_heatmaps

    def visualize_samples(self, n_samples=5):
        """
        Visualize sample images with their attribution heatmaps.

        Args:
            n_samples: Number of samples to visualize from each group
        """
        # Load heatmaps
        heatmap_path0 = os.path.join(self.heatmap_dir, f"group0_{self.args.explanation_method}_heatmaps.npy")
        heatmap_path1 = os.path.join(self.heatmap_dir, f"group1_{self.args.explanation_method}_heatmaps.npy")

        if not os.path.exists(heatmap_path0) or not os.path.exists(heatmap_path1):
            print("Heatmaps not found. Run the main processing first.")
            return

        group0_heatmaps = np.load(heatmap_path0)
        group1_heatmaps = np.load(heatmap_path1)

        # Visualize samples from each group
        for group_idx, (images, heatmaps, group_name) in enumerate(
            [(self.group0_np, group0_heatmaps, "group0"), (self.group1_np, group1_heatmaps, "group1")]
        ):
            print(f"\nVisualizing {n_samples} samples from {group_name}")

            for i in range(min(n_samples, len(images))):
                image = images[i]
                heatmap = heatmaps[i]

                # Create overlay
                overlaid = self.overlay_heatmap(image, heatmap, alpha=self.args.alpha)

                # Save overlay
                overlay_path = os.path.join(
                    self.overlay_dir, f"{group_name}_sample_{i}_{self.args.explanation_method}.png"
                )
                Image.fromarray(overlaid).save(overlay_path)

            print(f"  Saved overlays to {self.overlay_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Classifier Explanation with Multiple Methods")
    parser.add_argument("--n", type=int, default=100, help="Number of samples in group 0")
    parser.add_argument("--m", type=int, default=100, help="Number of samples in group 1")
    parser.add_argument("--bs", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--dst", type=str, default="corr", help="Dataset type (only 'corr' supported in this script)")
    parser.add_argument("--deg", type=str, default="zer-test", help="Corruption type (only 'zer-test' supported)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model_path",
        type=str,
        default="adni_results/ckps/model_finetun_last_7_True.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--explanation_method",
        type=str,
        default="lrp",
        help="Explanation method to use (e.g., 'lrp', 'guided_backprop', 'input_x_gradient')",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for overlay visualization")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization overlays after processing")
    parser.add_argument("--n_vis", type=int, default=5, help="Number of samples to visualize from each group")

    args = parser.parse_args()

    # Create experiment instance
    experiment = SupervisedExplainer(args)

    # Run attribution
    group0_heatmaps, group1_heatmaps = experiment.run()

    # Optionally visualize samples
    if args.visualize:
        experiment.visualize_samples(n_samples=args.n_vis)

    print("\n" + "=" * 50)
    print("Processing complete!")
    print("=" * 50)
