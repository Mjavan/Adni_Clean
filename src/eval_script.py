#!/usr/bin/env python


from pathlib import Path
import sys
import argparse
from argparse import Namespace

from src.eval_utils import plot_faithfulness_results, compute_auroc_faithfulness, print_auroc_table

# Add src directory to Python path
project_root = Path(__file__).parent.absolute()
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.vis2samtestdr import TestStatisticBackprop

# Base arguments
base_args = {
    'annot_path': '/sc/projects/sci-lippert/chair/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256/group_by_hippocampus/adni_T1_3T_linear_annotation.csv',
    'sav_gr_np': False,
    'corrupted': False,
    'deg': 'None',
    'ckp': 'fnt',
    'img_path': '/sc/projects/sci-lippert/chair/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256/group_by_hippocampus',
    'n': 100, 'm': 100, 'bs': 10,
    'dst': 'faithfulness_eval',
    'idx': 8,
    'random_state': 42,
    'model_path': 'adni_results/ckps/model_finetun_last_10_False.pt',
    'target_layer': '0.7.2.conv3',
    # Faithfulness evaluation parameters
    'n_superpixels': 10,
    'superpixel_compactness': 10.0,
    'circle_radius': 20,
    'circle_center_offset': -20,
    'circle_grey_value': 128,
    # LRP-specific parameters (optional, with defaults)
    'lrp_composite': 'epsilon_plus_flat',
    'lrp_epsilon': 1e-6,
    'lrp_gamma': 0.25,
    'lrp_input_low': 0.0,
    'lrp_input_high': 1.0
}

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run faithfulness evaluation for explainability methods')
    parser.add_argument(
        '--methods',
        nargs='+',
        default=["cam", "cam++", "lcam", "lrp"],
        choices=["cam", "cam++", "lcam", "lrp"],
        help='Explainability methods to evaluate (default: cam cam++ lcam lrp)'
    )
    cli_args = parser.parse_args()

    # Methods to evaluate
    METHODS = cli_args.methods
    print(f"Evaluating methods: {', '.join([m.upper() for m in METHODS])}")

    # Dictionary to store results
    all_results = {}

    # Run for each method
    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"Processing: {method.upper()}")
        print(f"{'='*60}")

        # Create args for this method
        args = Namespace(**base_args)
        args.expl = method
        args.exp = f'{method}-faithfulness-test-eval'

        # Run experiment
        experiment = TestStatisticBackprop(args)
        group0_attr, group1_attr = experiment.run()
        results = experiment.faithfulness_eval()

        # Store results
        all_results[method] = results

    all_results["random"] = experiment.faithfulness_eval(random_attr=True)
    # Plot all results together
    plot_faithfulness_results(all_results)

    # Compute and print AUROC
    auroc_dict = compute_auroc_faithfulness(all_results)
    print_auroc_table(auroc_dict)
    