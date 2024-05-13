import logging
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def save_figures(target, features, figures_path: Path) -> None:
    """Generate statistics and visualizations for summarizing the data and save to disk.

    Args:
        features (pd.DataFrame): The features DataFrame.
        figures_path (Path): The path to save the figures.
    """
    # Create the figures directory if it doesn't exist
    figures_path.mkdir(parents=True, exist_ok=True)

    figs = []

    for feat in features.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.hist([
            features[target == 0][feat].values, features[target == 1][feat].values
        ])
        ax.set_xlabel(' '.join(feat.split('_')).capitalize())
        ax.set_ylabel('Number of observations')
        figs.append(fig)

    # Save each figure to disk
    for i, fig in enumerate(figs):
        fig_path = figures_path / f'figure_{i+1}.png'
        fig.savefig(fig_path)
        plt.close(fig)
        logger.info("Figure saved to: %s", fig_path)
