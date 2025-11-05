import copy
from typing import Any

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
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.eval_utils import insert_grey_circle, segment_image_with_circle_superpixel


class TestStatisticBackprop:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._load_checkpoint()
        self._setup_experiment()
        self.heatmap_path = {}
        self.heatmap_path["gr1"] =  os.path.join(
            self.heatmap_dir, f"gr1_{len(self.group_1)}_{self.m1}_{self.args.expl}_{self.args.exp}.npy"
        )
        self.heatmap_path["gr2"] = os.path.join(
            self.heatmap_dir, f"gr2_{len(self.group_2)}_{self.m2}_{self.args.expl}_{self.args.exp}.npy"
        )

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
            # root_dir = "/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability"
            root_dir = "."
            #test_dir = Path(root_dir) / "AdniGithub" / "adni_results" / "split" / "test" / "False" / "None"
            test_dir = Path(root_dir) / "adni_results" / "split" / "test" / "False" / "None"
            out_path = test_dir / "test_split.npz"
            with np.load(out_path) as f:
                print(f.files)  # -> ['test0', 'test1']
                self.group_1 = f["test0"]
                self.group_2 = f["test1"]
            self.group_1_np = self.group_1[: self.args.n]
            self.group_2_np = self.group_2[: self.args.m]

        elif self.args.dst == "faithfulness_eval":
            print(f"Using test set for getting embeddings")
            # root_dir = "/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability"
            root_dir = "."
            #test_dir = Path(root_dir) / "AdniGithub" / "adni_results" / "split" / "test" / "False" / "None"
            test_dir = Path(root_dir) / "adni_results" / "split" / "test" / "False" / "None"
            out_path = test_dir / "test_split.npz"
            with np.load(out_path) as f:
                print(f.files)  # -> ['test0', 'test1']
                self.group_1 = f["test0"]
                self.group_2 = copy.deepcopy(f["test0"])
            self.group_1_np = self.group_1[: self.args.n]
            self.group_2_np = self.group_2[: self.args.m].copy()

            self.group_2_np = self._add_grey_circle_artifact_all_samples(self.group_2_np)

        elif self.args.dst == "full":
            dataset = AdniMRIDataset2D(annotations_file=self.args.annot_path, img_dir=self.args.img_path)
            self.group_1, self.group_2 = [], []
            for img, label in dataset:
                if label == 0:
                    self.group_1.append(img)
                else:
                    self.group_2.append(img)
            # concatenate list of arrays to a single numpy array
            self.group_1 = np.concatenate(self.group_1, axis=0)
            self.group_2 = np.concatenate(self.group_2, axis=0)
            # save two gropus as numpy arrays
            if self.args.sav_gr_np:
                full_path1 = os.path.join(self.img_dir, f"gr0_{len(self.group_1)}.npy")
                full_path2 = os.path.join(self.img_dir, f"gr1_{len(self.group_2)}.npy")
                np.save(full_path1, self.group_1)
                np.save(full_path2, self.group_2)
                print(f"group_1 shape:{self.group_1.shape}, group_2 shape:{self.group_2.shape}")
                print(f"group_1 and group_2 numpy arrays were saved!")
            # choose n,m samples from ecah group
            self.group_1_np = self.group_1[: self.args.n]
            self.group_2_np = self.group_2[: self.args.m]

        elif self.args.dst == "corr":

            if self.args.deg == "bl-test":
                print(f"Using corrupted test set for getting embeddings")
                # root_dir = "/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability"
                root_dir = "."
                test_dir = Path(root_dir) / "AdniGithub" / "adni_results" / "split" / "test" / "True"
                out_path = test_dir / "test_split.npz"
                with np.load(out_path) as f:
                    print(f.files)  # -> ['test0', 'test1']
                    self.group_1 = f["test0"]
                    self.group_2 = f["test1"]
                self.group_1_np = self.group_1[: self.args.n]
                self.group_2_np = self.group_2[: self.args.m]

            if self.args.deg == "zer-test":
                print(f"Using corrupted test set for getting embeddings")
                # root_dir = "/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability"
                root_dir = "."
                test_dir = Path(root_dir) / "adni_results" / "split" / "test" / "True"
                out_path = test_dir / "zer32" / "test_split.npz"
                print("images with patch size 32 corrupted are used")
                with np.load(out_path) as f:
                    print(f.files)  # -> ['test0', 'test1']
                    self.group_1 = f["test0"]
                    self.group_2 = f["test1"]

                self.group_1_np = self.group_1[: self.args.n]
                self.group_2_np = self.group_2[: self.args.m]

        # convert numpy array to tensor
        self.group_1 = self._convert_to_tensor(self.group_1_np)
        self.group_2 = self._convert_to_tensor(self.group_2_np)
        # make dataloaders for each group
        self.group_1_loader = DataLoader(self.group_1, batch_size=self.args.bs, shuffle=False, drop_last=True)
        self.group_2_loader = DataLoader(self.group_2, batch_size=self.args.bs, shuffle=False, drop_last=True)
        # save_attributions(group_1_attr, group_2_attr,latent_dim_idx)
        self.m1 = self.group_1.shape[0]
        self.m2 = self.group_2.shape[0]
        print(f"Data Loaders were built!")
        print(f"########################")

    def _add_grey_circle_artifact_all_samples(self, group_np):
        # Apply grey circle to each sample in group_2
        print(f"Applying grey circle to each sample in group_2...")
        for i in range(len(group_np)):
            # Get image dimensions
            height, width = group_np[i].shape
            # Insert grey circle at center with specified radius and offset
            center_y = height // 2 + self.args.circle_center_offset
            center_x = width // 2 + self.args.circle_center_offset
            center = (center_y, center_x)
            group_np[i] = insert_grey_circle(group_np[i], center, self.args.circle_radius, self.args.circle_grey_value)

        return group_np

    def _load_checkpoint(self):
        """Load model checkpoint."""
        # This is the path for self-supervised SimCLR model
        if self.args.ckp == "simclr":
            print("Using self-supervised pre-trained model")
            # base_path = "/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability/AdniGithub"
            base_path = "."
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

        elif self.args.ckp == "fnt":
            print("Using fine-tuned model on two groups of data without corruption (False)")
            # base_path = "/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability/AdniGithub"
            base_path = "."
            root_dir = Path(base_path)
            # checkpoint_dir = root_dir / 'adni_results' / 'ckps' / 'model_finetun_last_2_False.pt'
            checkpoint_dir = os.path.join(root_dir, self.args.model_path)
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

        elif self.args.ckp == "fnt_bl":
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

        elif self.args.ckp == "fnt_zer":
            print("Using fine-tuned model on two groups of data with corruption (True)")
            # base_path = "/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability/AdniGithub"
            base_path = "."
            root_dir = Path(base_path)
            checkpoint_dir = os.path.join(root_dir, self.args.model_path)
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

        elif self.args.ckp == "suppr":
            print("Using supervised pre-trained model without fine-tuning")
            # base_path = "/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability/AdniGithub"
            base_path = "."
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
            explainer = GradCAM(self.encoder, target_layer=self.args.target_layer, relu=False, device=self.device)
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
        group_1_mean = self.get_mean_embeddings(explainer, self.group_1_loader)
        group_2_mean = self.get_mean_embeddings(explainer, self.group_2_loader)
        D = group_1_mean - group_2_mean
        print(f"group_1_mean:{group_1_mean.shape}")
        print(f"group_2_mean:{group_2_mean.shape}")
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
            D: Difference vector (group_1_mean - group_2_mean)
            group_id: 0 for group_1, 1 for group_2
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

                # Generate attribution and move to CPU immediately
                # attributions = explainer.generate(flip_sign=(group_id == 1)) -- THIS KILLS THE EVALS
                attributions = explainer.generate()
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
        group_1_attr, group_1_embed = self.process_attributions(
            self.group_1_loader, explainer, D=D, group_id=0, use_squared=use_squared
        )
        group_2_attr, group_2_embed = self.process_attributions(
            self.group_2_loader, explainer, D=D, group_id=1, use_squared=use_squared
        )

        # compute test-statistic and p-value
        mmd = MMDTest(features_X=group_1_embed, features_Y=group_2_embed, n_perm=1000)
        test_statistic = mmd._compute_mmd(group_1_embed, group_2_embed)
        p_value = mmd._compute_p_value()
        print(f"Test statistic (MMD): {test_statistic:.4f}, p-value: {p_value:.4f}")

        # save_attributions(group_1_attr, group_2_attr,latent_dim_idx)
        full_path1 = self.heatmap_path["gr1"]
        full_path2 = self.heatmap_path["gr2"]

        np.save(full_path1, group_1_attr)
        np.save(full_path2, group_2_attr)

        print(f"gr1:{group_1_attr.shape}")
        print(f"gr2:{group_2_attr.shape}")

        print("Heatmaps were created")

        return (group_1_attr, group_2_attr)

    def overlay_hetmap(self, idx, alpha=0.5):
        """Overlay heatmap on the original image."""
        # Load original image
        if idx < len(self.group_1):
            img0 = self.group_1_np[idx]
        if idx < len(self.group_2):
            img1 = self.group_2_np[idx]

        # Load heatmap
        full_path1 = self.heatmap_path["gr1"]
        full_path2 = self.heatmap_path["gr2"]
        group_1_attr = np.load(full_path1)
        group_2_attr = np.load(full_path2)
        # Select corresponding heatmaps
        gcam_1 = group_1_attr[idx]
        gcam_2 = group_2_attr[idx]
        # Overlay heatmap
        _, overlaid_img0 = save_cam_with_alpha(img0, gcam_1, alpha=alpha)
        _, overlaid_img1 = save_cam_with_alpha(img1, gcam_2, alpha=alpha)

        # Save overlay images
        full_path1_ov = os.path.join(
            self.overlay_dir, f"gr1_{len(self.group_1)}_{self.m1}_{self.args.expl}_{self.args.idx}_{self.args.exp}.png"
        )
        full_path2_ov = os.path.join(
            self.overlay_dir, f"gr2_{len(self.group_2)}_{self.m2}_{self.args.expl}_{self.args.idx}_{self.args.exp}.png"
        )
        Image.fromarray(overlaid_img0).save(full_path1_ov)
        Image.fromarray(overlaid_img1).save(full_path2_ov)
        return overlaid_img0, overlaid_img1
    
    def faithfulness_eval(self, random_attr=False):
        """Evaluate faithfulness of attributions using superpixel-based ranking.

        Creates superpixel segmentation for each sample in group_2,
        then ranks all segments across all images by the sum of attributions
        within each segment. Then progressively replaces top-ranked superpixels
        with their group_1 counterparts and tracks changes in test statistic.

        Returns:
        --------
        results : dict
            Dictionary containing:
            - 'ranking_list': list of ranked segments
            - 'replacements': list of replacement steps with statistics
            - 'test_statistics': list of test statistics after each replacement
            - 'p_values': list of p-values after each replacement
        """
        if self.args.dst != "faithfulness_eval":
            raise ValueError("Faithfulness evaluation requires dst to be 'faithfulness_eval'")

        print("Starting faithfulness evaluation...")
        print("=" * 60)

        if random_attr == False:
            # Load heatmaps (attributions) for group_2
            print(f"Loading heatmaps from:")
            print(f"  Group 1: {self.heatmap_path['gr2']}")

            if not os.path.exists(self.heatmap_path['gr2']):
                raise FileNotFoundError(f"Heatmap file not found: {self.heatmap_path['gr2']}")

            group_2_heatmaps = np.load(self.heatmap_path['gr2'])  # Shape: (m_samples, height, width)

            print(f"Loaded heatmaps - Group 1: {group_2_heatmaps.shape}")
            print("=" * 60)

            # Step 1: Rank superpixels by attribution sum
            print("\nStep 1: Ranking superpixels by attribution sum...")
            ranking_list, superpixel_masks = self._rank_superpixels_by_attr_sum(self.group_2_np, group_2_heatmaps)
        else:
            print("\nRandomly shuffling superpixel rankings for control experiment...")
            ranking_list, superpixel_masks = self._rank_superpixels_by_attr_sum(self.group_2_np, np.random.rand(*self.group_2_np.shape))
            random.shuffle(ranking_list)

        # Step 2: Progressively replace superpixels and compute statistics
        print("\nStep 2: Progressively replacing superpixels and computing statistics...")
        results = self._progressive_replacement_analysis(self.group_2_np, self.group_1_np, ranking_list, superpixel_masks)

        return results

    def _rank_superpixels_by_attr_sum(self, images, heatmaps):
        """Rank all superpixels by their attribution sum.

        Returns:
        --------
        ranking_list : list of dict
            Ranked segments
        superpixel_masks : dict
            Dictionary mapping (image_idx, segment_id) to boolean mask
        """
        all_segments = []
        superpixel_masks = {}

        print(f"\nProcessing Group 1 ({len(images)} images)...")
        for img_idx in range(len(images)):
            image = images[img_idx]
            heatmap = heatmaps[img_idx]

            # Create superpixel segmentation
            center_y = image.shape[0] // 2 + self.args.circle_center_offset
            center_x = image.shape[1] // 2 + self.args.circle_center_offset
            superpixel_labels, _ = segment_image_with_circle_superpixel(
                image,
                center=(center_y, center_x),
                radius=self.args.circle_radius * 1.1, # increase radius slightly to cover full circle
                n_segments=self.args.n_superpixels,
                compactness=self.args.superpixel_compactness,
                visualize=False
            )

            # For each superpixel, calculate the sum of attributions
            unique_segments = np.unique(superpixel_labels)
            for segment_id in unique_segments:
                segment_mask = (superpixel_labels == segment_id)
                attribution_sum = np.sum(heatmap[segment_mask])

                # Store mask for later use
                superpixel_masks[(img_idx, int(segment_id))] = segment_mask

                all_segments.append({
                    'image_idx': img_idx,
                    'segment_id': int(segment_id),
                    'attribution_sum': float(attribution_sum),
                    'n_pixels': int(np.sum(segment_mask))
                })

            if (img_idx + 1) % 10 == 0:
                print(f"  Processed {img_idx + 1}/{len(images)} images")

        print(f"Group 1 complete: {len(all_segments)} segments total")

        # Sort all segments by attribution sum (descending)
        ranking_list = sorted(all_segments, key=lambda x: x['attribution_sum'], reverse=True)

        # Add rank to each segment
        for rank, segment in enumerate(ranking_list, start=1):
            segment['rank'] = rank

        return ranking_list, superpixel_masks

    def _progressive_replacement_analysis(self, group_np, group_to_replace_np, ranking_list, superpixel_masks):
        """Progressively replace top-ranked superpixels and compute statistics.

        Parameters:
        -----------
        ranking_list : list of dict
            Ranked segments
        superpixel_masks : dict
            Dictionary mapping (image_idx, segment_id) to boolean mask

        Returns:
        --------
        results : dict
            Analysis results with statistics for each replacement step
        """
        # Make a copy of group_2 that we'll progressively modify
        group_modified = group_np.copy()

        # Storage for results
        test_statistics = []
        p_values = []
        n_replaced = []

        # Convert to tensors
        group_to_replace_np_tensor = self._convert_to_tensor(group_to_replace_np)
        group_modified_tensor = self._convert_to_tensor(group_modified)

        # Create dataloaders
        group_to_replace_loader = DataLoader(group_to_replace_np_tensor, batch_size=self.args.bs, shuffle=False, drop_last=False)
        group_modified_loader = DataLoader(group_modified_tensor, batch_size=self.args.bs, shuffle=False, drop_last=False)
        group_to_replace_embed, group_modified_embed = self._retrieve_embeddings(group_to_replace_loader, group_modified_loader)

        group_to_replace_embed = np.vstack(group_to_replace_embed)
        group_modified_embed = np.vstack(group_modified_embed)

        # Compute baseline (no replacements)
        print("\nComputing baseline statistics (no replacements)...")
        test_stat, p_val = self._compute_test_statistic(group_to_replace_embed, group_modified_embed)
        test_statistics.append(float(test_stat))
        p_values.append(float(p_val))
        n_replaced.append(0)
        print(f"  Baseline: test_stat={test_stat:.4f}, p_value={p_val:.4f}")

        # Progressively replace superpixels
        replacement_steps = len(ranking_list)
        print(f"\nReplacing top {replacement_steps} superpixels...")

        for i in range(replacement_steps):
            segment = ranking_list[i]
            img_idx = segment['image_idx']
            segment_id = segment['segment_id']

            # Get the mask for this superpixel
            mask = superpixel_masks[(img_idx, segment_id)]

            # Replace pixels in group_2_modified with corresponding pixels from group_1
            group_modified[img_idx][mask] = group_to_replace_np[img_idx][mask]
            step_embeddings = self.encoder(self._convert_to_tensor(np.expand_dims(group_modified[img_idx],0)))
            step_embeddings = step_embeddings.view(step_embeddings.size()[0], -1)
            group_modified_embed[img_idx] = step_embeddings.detach().cpu().numpy()

            # Compute statistics after this replacement
            if (i + 1) % 10 == 0 or i < 10 or i == replacement_steps - 1:
                test_stat, p_val = self._compute_test_statistic(group_to_replace_embed, group_modified_embed)
                test_statistics.append(float(test_stat))
                p_values.append(float(p_val))
                n_replaced.append(i + 1)
                print(f"  After {i+1} replacements: test_stat={test_stat:.4f}, p_value={p_val:.4f}")

        results = {
            'ranking_list': ranking_list,
            'test_statistics': test_statistics,
            'p_values': p_values,
            'n_replaced': n_replaced
        }

        return results

    def _compute_test_statistic(self, group_1_embed, group_2_embed):
        """Compute MMD test statistic and p-value for two groups of images.

        Parameters:
        -----------
        group_1_images : np.ndarray
            Group 0 images (n_samples, height, width)
        group_2_images : np.ndarray
            Group 1 images (m_samples, height, width)

        Returns:
        --------
        test_statistic : float
            MMD test statistic
        p_value : float
            P-value from permutation test
        """

        # compute test-statistic and p-value
        mmd = MMDTest(features_X=group_1_embed, features_Y=group_2_embed, n_perm=1000)
        test_statistic = mmd._compute_mmd(group_1_embed, group_2_embed)
        p_value = mmd._compute_p_value()
        print(f"Test statistic (MMD): {test_statistic:.4f}, p-value: {p_value:.4f}")

        return test_statistic, p_value

    def _retrieve_embeddings(self, group_1_loader: DataLoader[Any], group_2_loader: DataLoader[Any]):

        group_1_embed = []
        group_2_embed = []


        with torch.no_grad():
            for images in group_1_loader:
                images = images.to(self.device)
                embeddings = self.encoder(images)
                embeddings = embeddings.view(embeddings.size()[0], -1)
                group_1_embed.append(embeddings.cpu().numpy())

            for images in group_2_loader:
                images = images.to(self.device)
                embeddings = self.encoder(images)
                embeddings = embeddings.view(embeddings.size()[0], -1)
                group_2_embed.append(embeddings.cpu().numpy())

        return group_1_embed, group_2_embed


    def max_sensitivity_evaluation(self, n_samples=50, lower_bound=-0.05, upper_bound=0.05, norm_ord=2):
        """Evaluate max-sensitivity of attributions.

        Measures robustness by perturbing ALL samples in both groups,
        recalculating test statistics, and measuring changes.
        Based on Yeh et al. (2019) "On the (In)fidelity and Sensitivity of Explanations".

        Parameters:
        -----------
        n_samples : int
            Number of perturbation iterations (default: 50)
        lower_bound : float
            Lower bound for uniform noise perturbation (default: -0.05)
        upper_bound : float
            Upper bound for uniform noise perturbation (default: 0.05)
        norm_ord : int
            Order of the norm for sensitivity computation (default: 2 for Frobenius)

        Returns:
        --------
        results : dict
            Dictionary containing sensitivity results.
        """
        if self.args.dst != "test":
            raise ValueError("Max-sensitivity evaluation requires dst to be 'test'")

        print("Starting max-sensitivity evaluation...")
        print("=" * 60)

        # Get original embeddings for both groups
        print("Computing baseline embeddings...")
        group_1_tensor = self._convert_to_tensor(self.group_1_np)
        group_2_tensor = self._convert_to_tensor(self.group_2_np)

        group_1_loader = DataLoader(group_1_tensor, batch_size=self.args.bs, shuffle=False, drop_last=False)
        group_2_loader = DataLoader(group_2_tensor, batch_size=self.args.bs, shuffle=False, drop_last=False)

        group_1_embed_original, group_2_embed_original = self._retrieve_embeddings(group_1_loader, group_2_loader)
        group_1_embed_original = np.vstack(group_1_embed_original)
        group_2_embed_original = np.vstack(group_2_embed_original)

        # Compute baseline test statistic and p-value
        original_test_stat, original_p_value = self._compute_test_statistic(group_1_embed_original, group_2_embed_original)
        print(f"Baseline test_stat: {original_test_stat:.4f}, p_value: {original_p_value:.4f}")

        # Compute baseline attributions
        print("Computing baseline attributions...")
        _, D, explainer = self.backprobagate_statistics()
        group_1_attr_original, _ = self.process_attributions(
            group_1_loader, explainer, D=D, group_id=0, use_squared=True
        )
        group_2_attr_original, _ = self.process_attributions(
            group_2_loader, explainer, D=D, group_id=1, use_squared=True
        )
        print("=" * 60)

        # Storage for sensitivities
        test_stat_sensitivities = []
        p_value_sensitivities = []
        attribution_sensitivities_group1 = []
        attribution_sensitivities_group2 = []

        # Perform n_samples perturbation iterations
        print(f"\nPerforming {n_samples} perturbation iterations on all samples...")

        for iteration in range(n_samples):
            # Add noise directly to tensors (values are already ImageNet normalized)
            noise_1 = torch.FloatTensor(group_1_tensor.shape).uniform_(lower_bound, upper_bound)
            group_1_perturbed_tensor = group_1_tensor + noise_1.to(self.device)

            noise_2 = torch.FloatTensor(group_2_tensor.shape).uniform_(lower_bound, upper_bound)
            group_2_perturbed_tensor = group_2_tensor + noise_2.to(self.device)

            group_1_perturbed_loader = DataLoader(group_1_perturbed_tensor, batch_size=self.args.bs, shuffle=False, drop_last=False)
            group_2_perturbed_loader = DataLoader(group_2_perturbed_tensor, batch_size=self.args.bs, shuffle=False, drop_last=False)

            group_1_embed_perturbed, group_2_embed_perturbed = self._retrieve_embeddings(group_1_perturbed_loader, group_2_perturbed_loader)
            group_1_embed_perturbed = np.vstack(group_1_embed_perturbed)
            group_2_embed_perturbed = np.vstack(group_2_embed_perturbed)

            # Compute perturbed test statistic and p-value
            perturbed_test_stat, perturbed_p_value = self._compute_test_statistic(group_1_embed_perturbed, group_2_embed_perturbed)

            # Compute D from perturbed embeddings
            group_1_mean_perturbed = torch.tensor(group_1_embed_perturbed.mean(axis=0)).to(self.device)
            group_2_mean_perturbed = torch.tensor(group_2_embed_perturbed.mean(axis=0)).to(self.device)
            D_perturbed = group_1_mean_perturbed - group_2_mean_perturbed

            # Compute perturbed attributions using perturbed D
            group_1_attr_perturbed, _ = self.process_attributions(
                group_1_perturbed_loader, explainer, D=D_perturbed, group_id=0, use_squared=True
            )
            group_2_attr_perturbed, _ = self.process_attributions(
                group_2_perturbed_loader, explainer, D=D_perturbed, group_id=1, use_squared=True
            )

            # Compute sensitivity for this iteration
            test_stat_diff = abs(original_test_stat - perturbed_test_stat)
            p_value_diff = abs(original_p_value - perturbed_p_value)

            # Compute MSE (Mean Squared Error) for attribution differences
            attr_diff_group1 = np.mean((group_1_attr_original - group_1_attr_perturbed) ** 2)
            attr_diff_group2 = np.mean((group_2_attr_original - group_2_attr_perturbed) ** 2)

            test_stat_sensitivities.append(test_stat_diff)
            p_value_sensitivities.append(p_value_diff)
            attribution_sensitivities_group1.append(attr_diff_group1)
            attribution_sensitivities_group2.append(attr_diff_group2)

            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"  Iteration {iteration + 1}/{n_samples}: "
                      f"test_stat_diff={test_stat_diff:.4f}, p_value_diff={p_value_diff:.4f}, "
                      f"attr_diff_g1={attr_diff_group1:.4f}, attr_diff_g2={attr_diff_group2:.4f}")

        # Compute statistics
        mean_test_stat_sensitivity = np.mean(test_stat_sensitivities)
        std_test_stat_sensitivity = np.std(test_stat_sensitivities)
        max_test_stat_sensitivity = np.max(test_stat_sensitivities)

        mean_p_value_sensitivity = np.mean(p_value_sensitivities)
        std_p_value_sensitivity = np.std(p_value_sensitivities)
        max_p_value_sensitivity = np.max(p_value_sensitivities)

        mean_attr_sensitivity_group1 = np.mean(attribution_sensitivities_group1)
        std_attr_sensitivity_group1 = np.std(attribution_sensitivities_group1)
        max_attr_sensitivity_group1 = np.max(attribution_sensitivities_group1)

        mean_attr_sensitivity_group2 = np.mean(attribution_sensitivities_group2)
        std_attr_sensitivity_group2 = np.std(attribution_sensitivities_group2)
        max_attr_sensitivity_group2 = np.max(attribution_sensitivities_group2)

        results = {
            'test_stat_sensitivities': test_stat_sensitivities,
            'p_value_sensitivities': p_value_sensitivities,
            'attribution_sensitivities_group1': attribution_sensitivities_group1,
            'attribution_sensitivities_group2': attribution_sensitivities_group2,
            'mean_test_stat_sensitivity': float(mean_test_stat_sensitivity),
            'std_test_stat_sensitivity': float(std_test_stat_sensitivity),
            'max_test_stat_sensitivity': float(max_test_stat_sensitivity),
            'mean_p_value_sensitivity': float(mean_p_value_sensitivity),
            'std_p_value_sensitivity': float(std_p_value_sensitivity),
            'max_p_value_sensitivity': float(max_p_value_sensitivity),
            'mean_attr_sensitivity_group1': float(mean_attr_sensitivity_group1),
            'std_attr_sensitivity_group1': float(std_attr_sensitivity_group1),
            'max_attr_sensitivity_group1': float(max_attr_sensitivity_group1),
            'mean_attr_sensitivity_group2': float(mean_attr_sensitivity_group2),
            'std_attr_sensitivity_group2': float(std_attr_sensitivity_group2),
            'max_attr_sensitivity_group2': float(max_attr_sensitivity_group2),
            'n_samples': n_samples,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'norm_ord': norm_ord,
            'original_test_stat': float(original_test_stat),
            'original_p_value': float(original_p_value)
        }


        return results
    
    
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

    # Faithfulness evaluation arguments
    parser.add_argument(
        "--n_superpixels",
        type=int,
        default=10,
        required=False,
        help="Number of superpixels for faithfulness evaluation. Lower values create larger, smoother superpixels (default: 50)",
    )
    parser.add_argument(
        "--superpixel_compactness",
        type=float,
        default=10.0,
        required=False,
        help="Compactness parameter for superpixel segmentation. Higher values (20-40) create more regular, compact shapes (default: 20.0)",
    )
    parser.add_argument(
        "--circle_radius",
        type=int,
        default=20,
        required=False,
        help="Radius of the circle region for faithfulness evaluation (default: 20)",
    )
    parser.add_argument(
        "--circle_center_offset",
        type=int,
        default=0,
        required=False,
        help="Offset for circle center from image center, applied to both x and y (default: 0)",
    )
    parser.add_argument(
        "--circle_grey_value",
        type=int,
        default=128,
        required=False,
        help="Grey value for the circle (0-255, default: 128 for mid-grey)",
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
    group_1_attr, group_2_attr = experiment.run()

    # Overlay heatmap on original image
    ov1, ov2 = experiment.overlay_hetmap(idx=args.idx, alpha=0.5)
