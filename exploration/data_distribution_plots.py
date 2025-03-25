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


def create_label_histogram_plot(amt_instances_per_class: list[int], weather_condition: str):
    # Data: The amount of instances in each class

    # Class names
    class_names = [
        'void', 'terrain', 'vegetation', 'lane marking', 'animal', 'bird', 'dynamic', 'ground', 
        'static', 'person', 'rider', 'pole', 'road', 'sidewalk', 'sky', 'building', 'fence', 
        'wall', 'traffic light', 'traffic sign', 'bicycle', 'bus', 'car', 'ego vehicle', 
        'motorcycle', 'truck', 'vehicle'
    ]

    assert(len(amt_instances_per_class) == len(class_names))

    # Plotting the histogram
    plt.figure(figsize=(6, 7))
    plt.barh(class_names, amt_instances_per_class, color='blue')

    # Set y-axis to logarithmic scale with a minimum value close to 0
    plt.xscale('log')

    # Adding labels and title
    plt.ylabel('Class Names')
    plt.xlabel('Amount of Instances (Log Scale)')
    plt.title(f'Validation Set Label Distribution in {weather_condition} Weather Conditions')

    # Displaying the plot
    plt.tight_layout()

    plt.savefig(f"imgs/data_distribution_plots/label_histogram_{weather_condition.lower()}_val_set.png", bbox_inches='tight', dpi=300)


def weather_conditions_in_splits_distribution_plot():
    # Data
    categories = ["Training", "Validation", "Test"]
    subcategories = ["Sunny", "Rain"]
    values = [52740, 230, 15060, 65, 7530, 30]

    # Define bar positions
    x = np.array([0, 0.2, 0.5, 0.7, 1.0, 1.2]) 

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))  # Compact figure
    ax.bar(x, values, width=0.15, color=['orange', 'blue', 'orange', 'blue', 'orange', 'blue'])

    # Labels and title
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_ylabel("Amount of Elements")
    ax.set_title("Distribution of Weather Conditions among Dataset Splits")
    ax.set_ylim(0, max(values) * 1.13)

    bar_label_vertical_distance = 2_000

    # Show values on bars
    for i, val in enumerate(values):
        ax.text(x[i], val + bar_label_vertical_distance, f"{val:,}", ha='center', fontsize=10)
    
    # Set x-axis labels for individual bars
    ax.set_xticks(x)
    x_labels = subcategories * len(categories)
    ax.set_xticklabels(x_labels, rotation=0)
    
    # Add grouped labels below x-axis labels
    group_x_positions = [(x[i] + x[i+1]) / 2 for i in range(0, len(x), 2)]
    #ax.set_xticks(list(x) + group_x_positions)
    #ax.set_xticklabels(x_labels)
    
    for i, label in enumerate(categories):
        ax.text(group_x_positions[i], -max(values) * 0.2, label, ha='center')
    
    # Reduce thickness of all lines
    ax.spines["top"].set_linewidth(0.5)
    ax.spines["right"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(width=0.5)  # Thinner tick marks

    # Adjust margins to ensure everything fits well
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.35, top=0.85)

    # Save the plot
    plt.savefig("imgs/data_distribution_plots/weather_conditions_among_dataset_splits.png", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    #create_waymo_images_distribution_plot()
    #create_waymo_point_clouds_distribution_plot()
    amt_instances_per_class_rain = [
        10950918, 0, 57868792, 7239604, 0, 948, 810007, 62050014, 104316999, 3012,
        131, 2212550, 179896348, 6404749, 203176894, 34706167, 0, 0, 69423, 490195,
        0, 1327672, 9333757, 108071, 0, 19025945, 385404]
    amt_instances_per_class_sun = [1111850103, 0, 31377100173, 1726251821,     1822698,      856398,
        601506926,  9666761923,  8583761190,   884195669,    42712861,  1534734284,
        28946861069, 10128116419, 19000459793, 34152743782,           0,           0,
        186467150,   540897836,    26805975,   718119579, 11852038228,    21285689,
        32634127,   915882531,   336761616]
    amt_instances_per_class_rain_val_set = [2243204, 0, 21095491, 1488030, 0, 0, 2099, 17181775, 18204670, 0, 0, 581121, 29529947, 844114, 40045962, 5318218, 0, 0, 10020, 87449, 0, 0, 3356570, 67326, 0, 19252, 272]
    create_label_histogram_plot(amt_instances_per_class_rain_val_set, "Rainy")
    #weather_conditions_in_splits_distribution_plot()