#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
import os



def mean_corrs(H):
    import itertools, numpy as np
    rs = []
    for i, j in itertools.combinations(range(len(H)), 2):
        rs.append(np.corrcoef(H[i].ravel(), H[j].ravel())[0,1])
    return np.mean(rs)



base = '.'
heatmap_dir = os.path.join(base, 'adni_results/heatmaps') 
n_sample = 100
m1 = 100

# name of explanation method
expl = 'cam'
# name of experiment to get heatmaps 
exp = 'cam-fnt-test'


path_heatmaps1 = os.path.join(heatmap_dir,f'gr1_{100}_{m1}_{expl}_{exp}.npy')
path_heatmaps2 = os.path.join(heatmap_dir,f'gr2_{100}_{m1}_{expl}_{exp}.npy')
heat_gr1_fnt_test = np.load(path_heatmaps1)
heat_gr2_fnt_test = np.load(path_heatmaps2)
print(heat_gr1_fnt_test.shape)
print(heat_gr2_fnt_test.shape)


r1 = mean_corrs(heat_gr1_fnt_test)

r2 = mean_corrs(heat_gr2_fnt_test)

print(f'r1:{r1:0.4f}, r2:{r2:0.4f}')



r_between = np.corrcoef(heat_gr1_fnt_test.mean(0).ravel(),
                        heat_gr2_fnt_test.mean(0).ravel())[0,1]

print(f'r_between: {r_between:0.4f}')



from pathlib import Path
import sys
from argparse import Namespace

# Add src directory to Python path
project_root = Path(__file__).parent.absolute()
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.vis2samtestdr import TestStatisticBackprop


args = Namespace(
    exp='cam-fnt-test-eval',
    annot_path='/sc/projects/sci-lippert/chair/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256/group_by_hippocampus/adni_T1_3T_linear_annotation.csv',
    sav_gr_np=False,
    corrupted=False,
    deg='None',
    ckp='fnt',
    expl='cam',
    img_path='/sc/projects/sci-lippert/chair/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256/group_by_hippocampus',
    n=100, m=100, bs=100,
    dst='test',
    idx=0,
    random_state=42,
    model_path='adni_results/ckps/model_finetun_last_10_False.pt',
    target_layer='0.7.2.conv3'
)

experiment = TestStatisticBackprop(args)
group0_attr, group1_attr = experiment.run()
ov1, ov2 = experiment.overlay_hetmap(idx=args.idx, alpha=0.5)

# Display the generated heatmaps and overlaid images
print(f'\nGenerated heatmap shapes:')
print(f'Group 0: {group0_attr.shape}')
print(f'Group 1: {group1_attr.shape}')

# Create a figure to display the overlaid images side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 7))

# Display Group 1 overlaid image
axes[0].imshow(ov1)
axes[0].set_title(f'Group 1 - Overlay (idx={args.idx})', fontsize=14)
axes[0].axis('off')

# Display Group 2 overlaid image
axes[1].imshow(ov2)
axes[1].set_title(f'Group 2 - Overlay (idx={args.idx})', fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.savefig('overlay_comparison.png', dpi=150, bbox_inches='tight')
print(f'\nOverlay comparison saved to: overlay_comparison.png')
plt.show()

