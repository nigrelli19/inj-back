import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from plot_lossmap_xc import load_multiple_lossmaps,  collimators_names_json

def main(base_dir):
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
    #collimators = collimators_names('lines_&_coll/CollDB_FCCee_z_tridodo572_wiggler_TCTs.dat') #collimators_names('lines_&_coll/CollDB_vertical_aperture_test.dat') #
    
    coll_types = ['tcp.h.', 'tcp.v.', 'tcp.hp','tcs.h.','tcs.v.', 'tcs.hp','tct.v', 'tct.h','tcr']  # Collimator types
    results = []


    single_file = f'merged_lossmap_full.json'
    collimator_df, aperture_df, from_part = load_multiple_lossmaps(base_dir, base_dir, single_file=single_file)
    
    if collimator_df.empty:
        print(f"No losses detected. Skipping.")

    # Filter for relevant collimators
    filtered_df = collimator_df[collimator_df['name'].isin(collimators)]
    
    if filtered_df.empty:
        print(f"No losses detected in relevant collimators for turn. Skipping.")
    
    # Group losses by collimator type
    for coll_type in coll_types:

        coll_filtered_df = filtered_df[filtered_df['name'].str.contains(coll_type, case=False)]
        
        if coll_filtered_df.empty:
            print(f"No losses detected for collimator type {coll_type.upper()}.")
            results.append({
                "Collimator Type": coll_type.upper(),
                "Total Loss": 0.0
            })
            continue

        coll_group = coll_filtered_df.groupby('name').agg(
            total_n=('n', 'sum'),
            s=('s', 'mean'),
            length=('length', 'mean')
        ).reset_index()
        
        # Calculate power lost and append results
        # total_loss = coll_group['total_n'].sum() * POW_TOT / norm
        total_loss = coll_group['total_n'].sum() * 100/ norm
        results.append({
            "Collimator Type": coll_type.upper(),
            "Total Loss": total_loss
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
    plt.bar(collimator_types, df_results['Total Loss'], color=colors)

    # Create legend handles
    legend_patches = [mpatches.Patch(color=colors[i], label=collimator_types.iloc[i]) for i in range(num_bars)]
    plt.legend(handles=legend_patches, title="Collimator Type", fontsize=12, title_fontsize=13, loc="upper left")

    # Formatting
    plt.xlabel("Collimator Type", fontsize=14)
    plt.ylabel("Total Loss [%]", fontsize=14)
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
    args = parser.parse_args()
    # Call the main function with parsed arguments
    main(args.base_dir)