import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from loss_functions import KL, calculate_iou
from helper import get_dm

matplotlib.use('pdf')


def plot_histogram_and_image(hist_pred, hist_true, img, tile_name, out_dir=None, volume_weighted=False):
    """
    Plotting the predicted histogram on the top of original histogram, 
    next to the image of the original tile.
    """
    img = img.astype(np.uint8)
    index = np.arange(len(hist_pred)) + 0.5

    # Metrics
    KL_div = KL(hist_true, hist_pred)
    iou = calculate_iou(hist_true, hist_pred)

    # Mean diameter calculation in cm
    dm_true = get_dm(hist_true, volume_weighted=volume_weighted)
    dm_pred = get_dm(hist_pred, volume_weighted=volume_weighted)

    # Create Figure and Axes instances
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6.5))

    # Add title
    fig.suptitle('Comparison of Distribution\n%s' % (tile_name), fontsize=18)

    ax1.bar(index, hist_true, width=1.0, label='true histogram')
    ax1.bar(index, hist_pred, width=1.0, alpha=0.5, label='predicted histogram')
    ax1.legend(fontsize=14)

    # Updated axis labels for your fine data
    ax1.set_xlabel('Grain diameter [mm]', fontsize=16)
    
    if volume_weighted:
        ax1.set_ylabel('Relative volume', fontsize=16)
    else:
        ax1.set_ylabel('Relative frequency', fontsize=16)

    # Dynamic x-axis labels matching your 0.05 - 80 mm log-spaced edges
    # Corrects the AttributeError by avoiding np.int
    edges_mm = np.logspace(np.log10(0.05), np.log10(80), 22)
    group_labels = [f"{x:.2f}" for x in edges_mm]
    
    ax1.set_xticks(np.arange(len(group_labels)))
    ax1.set_xticklabels(group_labels, rotation='vertical')

    # Performance annotation
    ax1.text(0.98, 0.82, 'KL: %.2f' % (KL_div), ha='right', va='top', transform=ax1.transAxes, fontsize=16)
    ax1.text(0.98, 0.76, 'IoU: %.2f' % (iou), ha='right', va='top', transform=ax1.transAxes, fontsize=16)
    ax1.text(0.98, 0.70, 'dm true: %.2f cm' % (dm_true), ha='right', va='top', transform=ax1.transAxes, fontsize=16)
    ax1.text(0.98, 0.64, 'dm pred: %.2f cm' % (dm_pred), ha='right', va='top', transform=ax1.transAxes, fontsize=16)

    ax2.set_xticks(())
    ax2.set_yticks(())
    ax2.imshow(img)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    if out_dir is not None:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        plt.savefig(os.path.join(out_dir, '{}.png'.format(tile_name)), bbox_inches='tight')
        plt.close(fig)