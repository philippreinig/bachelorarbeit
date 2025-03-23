import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator

def create_waymo_images_distribution_plot():
    # Data
    categories = ["Total", "Sunny", "Rain", "Sunny w/\nSegmentation Masks", "Rain w/\nSegmentation Masks"]
    values = [990340, 984375, 5965, 75355, 325]

    assert(values[0] == values[1] + values[2])

    # Define bar positions (reduce spacing between groups)
    x = np.array([0, 0.3, 0.5, 0.8, 1.0])  # Tighter grouping

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))  # Compact figure
    ax.bar(x, values, width=0.15, color=['green', 'orange', 'blue', 'orange', 'navy'])

    # Labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=90)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_ylabel("Number of Images")
    ax.set_title("Distribution of Images in WOD")
    ax.set_ylim(0, max(values) * 1.1)

    # Show values on bars
    for i, v in enumerate(values):
        ax.text(x[i], v + 30000, f"{v:,}", ha='center', fontsize=10)

    # Adjust y-axis ticks for better spacing
    ax.yaxis.set_major_locator(MultipleLocator(200000))

    # Reduce thickness of all lines
    ax.spines["top"].set_linewidth(0.5)
    ax.spines["right"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(width=0.5)  # Thinner tick marks

    # Adjust margins to ensure everything fits well
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.3, top=0.85)

    # Save the plot
    plt.savefig("imgs/data_distribution_plots/waymo_data_distribution_images.png", bbox_inches='tight', dpi=300)

from matplotlib.ticker import MultipleLocator

def create_waymo_point_clouds_distribution_plot():
    # Data
    categories = ["Total", "Sunny", "Rain", "Sunny w/\nSegmentation Labels", "Rain w/\nSegmentation Labels"]
    values = [989936, 983971, 5965, 29517, 150]

    assert(values[0] == values[1] + values[2])


    # Define bar positions (reduce spacing between groups)
    x = np.array([0, 0.3, 0.5, 0.8, 1.0])  # Tighter grouping

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))  # Compact figure
    ax.bar(x, values, width=0.15, color=['green', 'orange', 'blue', 'orange', 'navy'])

    # Labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=90)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_ylabel("Number of Point Clouds")
    ax.set_title("Distribution of Point Clouds in WOD")
    ax.set_ylim(0, max(values) * 1.1)

    # Show values on bars
    for i, v in enumerate(values):
        ax.text(x[i], v + 30000, f"{v:,}", ha='center', fontsize=10)

    # Adjust y-axis ticks for better spacing
    ax.yaxis.set_major_locator(MultipleLocator(200000))

    # Reduce thickness of all lines
    ax.spines["top"].set_linewidth(0.5)
    ax.spines["right"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(width=0.5)  # Thinner tick marks

    # Adjust margins to ensure everything fits well
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.3, top=0.85)

    # Save the plot
    plt.savefig("imgs/data_distribution_plots/waymo_data_distribution_point_clouds.png", bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    create_waymo_images_distribution_plot()
    create_waymo_point_clouds_distribution_plot()
