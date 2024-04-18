import matplotlib.pyplot as plt
import numpy as np
import sys

model_dt = 0.05

# Function to parse the data from the text file
def parse_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'): # Skip comment lines
                continue
            values = line.strip().split(',')
            data.append([float(value) for value in values])
    return data

# Function to plot the data
def plot_data(data):
    # Group data by k and j values
    groups = {}
    for entry in data:
        k, r, x = entry
        key = k
        if key not in groups:
            groups[key] = []
        groups[key].append(entry)

    print(f"Found {len(groups)} groups")

    # Create a figure with two subplots
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2)

    # Plot each group - first cmd_vel_vx
    colormap_viridis = plt.get_cmap('viridis')
    colormap_inferno = plt.get_cmap('Wistia')
    num_groups = len(groups)/2
    for index,(key, group) in enumerate(groups.items()):
        k = key
        r_values = [entry[1] for entry in group]
        x_values = [entry[2] for entry in group]
        x = np.array(x_values)
        v = np.gradient(x)/model_dt

        if key >= 25:
            color = colormap_viridis(index / num_groups) # Normalize index to colormap range
            ax1.plot(r_values, x, marker='o', color=color,
                     label=f'path_length={k}')

            ax3.plot(r_values, v, marker='o', color=color,
                     label=f'path_length={k}')
        else:
            color = colormap_inferno(index / num_groups) # Normalize index to colormap range
            ax2.plot(r_values, x, marker='o', color=color,
                     label=f'path_length={k}')

            ax4.plot(r_values, v, marker='o', color=color,
                     label=f'path_length={k}')


    ax1.set_xlabel(f'Iterations of optimizer->evalControl (model_dt={model_dt}, 4s)')
    ax1.set_ylabel('x')
    ax1.set_title('Plot of optimal trajectory x PathFollowCritic given varying path_length')

    ax1.legend()


    ax2.set_xlabel(f'Iterations of optimizer->evalControl (model_dt={model_dt}, 4s)')
    ax2.set_ylabel('x')
    ax2.set_title('Plot of optimal trajectory x GoalCritic given varying path_length')

    ax2.legend()
    #plt.axis([None, None, 15, 20])
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    data_file = sys.argv[1]
    data = parse_data(data_file)
    plot_data(data)

