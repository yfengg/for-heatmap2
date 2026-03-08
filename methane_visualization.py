import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load data
    data = pd.read_csv("methane_data.csv")
    data.columns = data.columns.str.strip()
    data["X"] = pd.to_numeric(data["X"], errors="coerce")
    data["Y"] = pd.to_numeric(data["Y"], errors="coerce")
    data["Baseline CH4 Production Rate"] = pd.to_numeric(data["Baseline CH4 Production Rate"], errors="coerce")
    data = data.dropna(subset=["X","Y","Baseline CH4 Production Rate"])

    print(f"Data loaded: {len(data)} rows")


    # Decide cube size (real units)
    cube_size = 50

    xmin, xmax = data["X"].min(), data["X"].max()
    ymin, ymax = data["Y"].min(), data["Y"].max()

    # Create bins for X and Y
    x_bins = np.arange(xmin, xmax + cube_size, cube_size)
    y_bins = np.arange(ymin, ymax + cube_size, cube_size)

    # Assign each point to a cube
    data["X_bin"] = pd.cut(data["X"], bins=x_bins, labels=False)
    data["Y_bin"] = pd.cut(data["Y"], bins=y_bins, labels=False)

    # Aggregate CH4 in each cube
    binned = data.groupby(["X_bin","Y_bin"])["Baseline CH4 Production Rate"].mean().reset_index()

    # Compute cube centers for plotting
    binned["X_center"] = xmin + binned["X_bin"] * cube_size + cube_size/2
    binned["Y_center"] = ymin + binned["Y_bin"] * cube_size + cube_size/2

    # Pivot table for heatmap
    heatmap = binned.pivot_table(
        values="Baseline CH4 Production Rate",
        index="Y_center",
        columns="X_center"
    )

    # Plot heatmap
    plt.figure(figsize=(12,10))
    sns.heatmap(
        heatmap,
        cmap="viridis",
        cbar_kws={"label":"CH4 Rate"},
        square=True
    )
    plt.gca().invert_yaxis()
    plt.title("Methane Heatmap (Aggregated Cubes)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()
    plt.savefig("methane_heatmap_aggregated_cubes.png", dpi=300)
    print("Saved methane_heatmap_aggregated_cubes.png")
    plt.close()

if __name__ == "__main__":
    main()