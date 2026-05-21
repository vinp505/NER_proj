import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Settings
# -----------------------------
CSV_PATH = "evaluation_ftmodel_eng.csv"

TARG_COL = "targ_lang"
TEST_COL = "test_lang"
GROUP_COL = "group"
EPOCH_COL = "epoch"
F1_COL = "F1"


for targets in [['eng','1'], ['dan', '1'], ['chi', '5'],['rom', '3'],['slk', '2'],['all', 'none']]:
    TARGET_LANG = targets[0]  # e.g. "eng", "dan", "chi", or None for no filtering
    TEST_LANG = None      # e.g. "eng", or None for no filtering

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    df[EPOCH_COL] = pd.to_numeric(df[EPOCH_COL])
    df[F1_COL] = pd.to_numeric(df[F1_COL])

    # Optional filtering
    plot_df = df.copy()

    if TARGET_LANG is not None:
        plot_df = plot_df[plot_df[TARG_COL] == TARGET_LANG]

    if TEST_LANG is not None:
        plot_df = plot_df[plot_df[TEST_COL] == TEST_LANG]

    # -----------------------------
    # Compute min/max/avg F1 per group and epoch
    # -----------------------------
    band_df = (
        plot_df
        .groupby([GROUP_COL, EPOCH_COL], as_index=False)[F1_COL]
        .agg(
            min_f1="min",
            max_f1="max",
            avg_f1="mean"
        )
    )

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    for group_name, group_data in band_df.groupby(GROUP_COL):
        group_data = group_data.sort_values(EPOCH_COL)

        x = group_data[EPOCH_COL].to_numpy()
        y_min = group_data["min_f1"].to_numpy()
        y_max = group_data["max_f1"].to_numpy()

        # Plot max line first just to get Matplotlib's automatic group color
        line, = ax.plot(
            x,
            y_max,
            linewidth=1.5,
            label=f"Group {group_name}"
        )

        color = line.get_color()

        # Plot min line with same color
        ax.plot(
            x,
            y_min,
            linewidth=1.2,
            linestyle="--",
            color=color,
            label="_nolegend_"
        )


        # Fill area between min and max
        ax.fill_between(
            x,
            y_min,
            y_max,
            color=color,
            alpha=0.20,
            linewidth=0
        )

    ax.set_title(f"F1 range by group" + (f" — target language: {TARGET_LANG}, group: {targets[1]}" if TARGET_LANG else ""))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Group")

    plt.tight_layout()
    plt.show()