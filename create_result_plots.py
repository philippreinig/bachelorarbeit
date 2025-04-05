import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

log = logging.getLogger(__name__)

def create_experiment_result_plot():
    # Data
    categories = ["No-Fusion", "Unimodal CLM-Fusion"]
    subcategories = ["ECE", "Macro-F1", "Accuracy"]
    values = [0.112, 0.371, 0.712, 0.178, 0.301, 0.668]	

    # Define bar positions
    x = np.array([0, 0.2, 0.4, 0.7, 0.9, 1.1]) 

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))  # Compact figure
    ax.bar(x, values, width=0.15, color=['salmon', 'yellowgreen', 'cornflowerblue', 'salmon', 'yellowgreen', 'cornflowerblue'])

    # Labels and title
    #ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_ylabel("Metric Values")
    ax.set_title(f"Unimodal CLM-Fusion in Combined Scenario")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)

    bar_label_vertical_distance = 0.05

    # Show values on bars
    for i, val in enumerate(values):
        ax.text(x[i], val + bar_label_vertical_distance, f"{val:,}", ha='center', fontsize=10)
    
    # Set x-axis labels for individual bars
    ax.set_xticks(x)
    x_labels = subcategories * len(categories)
    ax.set_xticklabels(x_labels, rotation=0)
    
    # Add grouped labels below x-axis labels
    group_x_positions = [x[i+1] for i in range(0, len(x), 3)]
    
    for i, label in enumerate(categories):
        ax.text(group_x_positions[i], -max(values) * 0.25, label, ha='center')
    
    # Reduce thickness of all lines
    ax.spines["top"].set_linewidth(0.5)
    ax.spines["right"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(width=0.5)  # Thinner tick marks

    # Adjust margins to ensure everything fits well
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.35, top=0.85)

    # Save the plot
    plt.savefig(f"imgs/result_plots/experiments/experiment_results_2.2.png", bbox_inches='tight', dpi=300)
    plt.show()

def create_model_result_plot():
    ece_sun = [0.004, 0.017, 0.011]
    f1_sun = [0, 0, 0]
    accuracy_sun = [1.00, 0.990, 0.997]

    ece_rain = [0.490, 0.790, 0.918]
    f1_rain = [0.667, 0.286, 0.124]
    accuracy_rain = [0.500, 0.167, 0.066]

    ece_combined = [0.255, 0.388, 0.419]
    f1_combined = [0.651, 0.186, 0.105]
    accuracy_combined = [0.746, 0.547, 0.556]    

    grid_labels = ["1x1", "2x2", "4x4"]
    x_values = np.arange(len(grid_labels))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.title("Weather Classifier")

    markersize=5
    accuracy_line_and_marker_style = "o-"
    f1_line_and_marker_style = "s--"
    ece_line_and_marker_style= "^:"
    sun_color = "y"
    rain_color = "b"
    combined_color = "g"

    # Line plots with different markers
    ax.plot(x_values, ece_sun, f"{sun_color}{ece_line_and_marker_style}", markersize=markersize, label="Sun ECE")
    ax.plot(x_values, f1_sun, f"{sun_color}{f1_line_and_marker_style}", markersize=markersize, label="Sun Macro-F1")
    ax.plot(x_values, accuracy_sun, f"{sun_color}{accuracy_line_and_marker_style}", markersize=markersize, label="Sun Accuracy")

    ax.plot(x_values, ece_rain, f"{rain_color}{ece_line_and_marker_style}", markersize=markersize, label="Rain ECE")
    ax.plot(x_values, f1_rain, f"{rain_color}{f1_line_and_marker_style}", markersize=markersize,label="Rain Macro-F1")
    ax.plot(x_values, accuracy_rain, f"{rain_color}{accuracy_line_and_marker_style}", markersize=markersize, label="Rain Accuracy")
    
    ax.plot(x_values, ece_combined, f"{combined_color}{ece_line_and_marker_style}", markersize=markersize,label="Combined ECE")
    ax.plot(x_values, f1_combined, f"{combined_color}{f1_line_and_marker_style}", markersize=markersize,label="Combined Macro-F1")
    ax.plot(x_values, accuracy_combined, f"{combined_color}{accuracy_line_and_marker_style}", markersize=markersize, label="Combined Accuracy")

    # X-axis labels
    ax.set_xticks(x_values, grid_labels)
    ax.set_yticks([i/10 for i in range(11)])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Labels and legend
    ax.set_xlabel("Grid")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0,1)
    #plt.legend()
    ax.grid(True, linestyle="-", axis='both', alpha=0.5)

    # Show plot
    plt.savefig("imgs/result_plots/models/wc_model_results.png", bbox_inches='tight', dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    #create_model_result_plot()

    create_experiment_result_plot()
