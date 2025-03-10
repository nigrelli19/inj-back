import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from plot_lossmap_xc import load_multiple_lossmaps,  collimators_names_json

def main(base_dir, n_part, percentage=False):
    # Initialize variables
    #turns = np.arange(0, 40)

    # Read the total loss value from the loss.txt file in the current base_dir
    loss_file = os.path.join(base_dir, "loss.txt")
    if not os.path.isfile(loss_file):
        raise FileNotFoundError(f"The file 'loss.txt' was not found in the directory: {base_dir}")
    
    with open(loss_file, "r") as file:
        lines = file.readlines()
    total_loss_line = next((line for line in lines if "Loss total" in line), None)
    if total_loss_line is None:
        raise ValueError(f"The 'Loss total' value is missing in the 'loss.txt' file.")

    # Extract the total loss value
    total_loss_str = total_loss_line.split(":")[1].strip()
    try:
        norm = float(total_loss_str)
    except ValueError:
        raise ValueError(f"Invalid 'Loss total' value in the 'loss.txt' file: {total_loss_str}")
    

    collimators = collimators_names_json('lines_&_coll/CollDB_FCCee_z_common_LSSs_TCS.json')

    coll_types = {
        'tcp.h.b1': 'tcp.h.b1',
        'tcp.v.b1': 'tcp.v.b1',
        'tcp.hp': 'tcp.hp',
        'tcs.h[12]': 'tcs.h',  # Matches only tcs.h1 and tcs.h2
        'tcs.v': 'tcs.v',
        'tcs.hp': 'tcs.hp',
        'tct.v': 'tct.v',
        'tct.h': 'tct.h',
        'tcr': 'tcr'
    }  # Collimator types
    results = []


    single_file = f'merged_lossmap_full.json'
    collimator_df, aperture_df, from_part = load_multiple_lossmaps(base_dir, base_dir, single_file=single_file)
    
    if collimator_df.empty:
        print(f"No losses detected. Skipping.")

    # Filter for relevant collimators
    filtered_df = collimator_df.query("name in @collimators")
    #filtered_df = collimator_df[collimator_df['name'].isin(collimators)]
    
    if filtered_df.empty:
        print(f"No losses detected in relevant collimators for turn. Skipping.")
    else:
        # Count total collimators of each type using the full collimators list
        collimator_counts = {category: sum(bool(re.match(pattern, name)) for name in collimators) 
                             for pattern, category in coll_types.items()}

        # Group losses by collimator type
        for pattern, category in coll_types.items():
            coll_filtered_df = filtered_df[filtered_df['name'].str.match(pattern, case=False, na=False)]
            num_collimators = collimator_counts[category]  # Get count from full collimator list

            if coll_filtered_df.empty or num_collimators == 0:
                print(f"No losses detected for collimator type {category.upper()}.")
                results.append({
                    "Collimator Type": category.upper(),
                    "Total Loss": 0.0,
                    "Normalized Loss": 0.0
                })
                continue

            coll_group = coll_filtered_df.groupby('name').agg(
                total_n=('n', 'sum'),
                s=('s', 'mean'),
                length=('length', 'mean')
            ).reset_index()

            if percentage:
                total_loss = coll_group['total_n'].sum() * 100 / norm
                total_loss /= num_collimators  # Normalize by total number of collimators
            else:
                norm = 45.6e9 * n_part / 17.5e4
                total_loss = coll_group['total_n'].sum() / norm

            results.append({
                "Collimator Type": category.upper(),
                "Total Loss": total_loss,
                "Normalized Loss": total_loss
            })

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    if df_results.empty:
        print("No losses detected across all turns. Exiting.")
    else:
        print(df_results['Total Loss'].sum())

    plt.figure(figsize=(8, 6))  # Adjust figure size

    # Use a colormap for distinct colors
    num_bars = len(df_results)
    colors = plt.cm.tab20.colors[:num_bars]  # Assign different colors
    collimator_types = df_results['Collimator Type']

    # Create bar plot
    plt.bar(collimator_types, df_results['Normalized Loss'], color=colors)

    # Create legend handles
    legend_patches = [mpatches.Patch(color=colors[i], label=collimator_types.iloc[i]) for i in range(num_bars)]
    plt.legend(handles=legend_patches, title="Collimator Type", fontsize=12, title_fontsize=13, loc="upper left")

    # Formatting
    plt.xlabel("Collimator Type", fontsize=14)
    if percentage:
        plt.ylabel("Total Loss [%]", fontsize=14)
    else:
        plt.ylabel("Total Loss [J]", fontsize=14)
    plt.title("Total Loss per Collimator Type", fontsize=16)
    plt.xticks(rotation=45, ha="right")  # Rotate labels for better readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save or Show Plot
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "collimator_losses_bar.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Track particles with a kicker and plot the results.')
    parser.add_argument('--base_dir', type=str, required=True, help='Path to lossmap files, for ex.: dataset/new_vertical/3_turns/ver_phase_90_3turns.')
    parser.add_argument('--n_part', type=int, required=True, help='Number of simulated particles')
    parser.add_argument('--percentage', action='store_true', help='Select True to plot percentage of energy lost in collimators')
    args = parser.parse_args()
    # Call the main function with parsed arguments
    main(args.base_dir, args.n_part, args.percentage)