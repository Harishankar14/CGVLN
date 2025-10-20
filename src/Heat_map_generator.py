

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def extract_lipschitz_data(report_file):
    """
    Parse vulnerability report and extract layer-by-layer Lipschitz constants
    Returns: (layer_names, lipschitz_values)
    """
    layers = []
    lipschitz = []
    
    with open(report_file, 'r') as f:
        for line in f:
            if 'Lipschitz Constant:' in line:
                # Extract the value
                value = float(line.split(':')[1].strip().replace('e+', 'e'))
                lipschitz.append(value)
            elif 'Node:' in line:
                # Extract layer name
                node = line.split(':')[1].strip()
                layers.append(node)
    
    return layers, lipschitz
'''
I yeah, i will hardcode the path.. 
'''
resnet_layers, resnet_lipschitz = extract_lipschitz_data('/home/harishankar/Music/cgvulnscan/Result/RESNET50.txt')
mobile_layers, mobile_lipschitz = extract_lipschitz_data('/home/harishankar/Music/cgvulnscan/Result/MobileNETV2.txt')
efficient_layers, efficient_lipschitz = extract_lipschitz_data('/home/harishankar/Music/cgvulnscan/Result/EfficientNET-V2.txt')


def categorize_lipschitz(value):
    if value < 10:
        return 0  # Safe (green)
    elif value < 50:
        return 1  # Medium (yellow)
    elif value < 100:
        return 2  # High (orange)
    elif value < 200:
        return 3  # Critical (red)
    else:
        return 4  # Extreme (dark red)

# Pad arrays to same length for visualization
max_layers = max(len(resnet_lipschitz), len(mobile_lipschitz), len(efficient_lipschitz))

resnet_padded = resnet_lipschitz + [0] * (max_layers - len(resnet_lipschitz))
mobile_padded = mobile_lipschitz + [0] * (max_layers - len(mobile_lipschitz))
efficient_padded = efficient_lipschitz + [0] * (max_layers - len(efficient_lipschitz))

# Categorize
resnet_cat = [categorize_lipschitz(x) for x in resnet_padded]
mobile_cat = [categorize_lipschitz(x) for x in mobile_padded]
efficient_cat = [categorize_lipschitz(x) for x in efficient_padded]

# Create heatmap
fig, ax = plt.subplots(figsize=(14, 6))

heatmap_data = np.array([resnet_cat, mobile_cat, efficient_cat])

# Custom colormap
colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
cmap = sns.color_palette(colors, as_cmap=True)

sns.heatmap(heatmap_data, 
            cmap=cmap,
            cbar_kws={'label': 'Vulnerability Severity', 
                      'ticks': [0, 1, 2, 3, 4]},
            yticklabels=['ResNet-50', 'MobileNetV2', 'EfficientNetV2'],
            xticklabels=False,
            linewidths=0.5,
            linecolor='white',
            ax=ax)

# Customize colorbar labels
colorbar = ax.collections[0].colorbar
colorbar.set_ticklabels(['Safe\n(<10)', 'Medium\n(10-50)', 'High\n(50-100)', 
                         'Critical\n(100-200)', 'Extreme\n(>200)'])

ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Architecture', fontsize=12, fontweight='bold')
ax.set_title('Layer-by-Layer Lipschitz Constant Vulnerability Heatmap', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('vulnerability_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('vulnerability_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("Heatmap saved as 'vulnerability_heatmap.pdf' and 'vulnerability_heatmap.png'")
