from tabulate import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from research.enums import ChartType

FIGURES_DIR = "research/figures/"


def check_figures_dir():
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)


def print_table(df: pd.DataFrame, precision: int = 2):

    print(
        "\n"
        + tabulate(
            df, headers="keys", tablefmt="heavy_grid", showindex=False, floatfmt=f".{precision}f"
        )
        + "\n"
    )


def chart(
    type: ChartType,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str = None,
    plot_frontier: bool = False,
    file_name: str = None,
    title: str = None,
    dimensions: tuple[int,int] = None
):
    # Capitalize Variables
    cols = [col for col in [x_col, y_col, z_col] if col]
    new_cols = [col.replace("_", " ").title() for col in cols]
    data = data.rename(columns={col: new_col for col, new_col in zip(cols, new_cols)})

    x_col, y_col = new_cols[:2]
    z_col = new_cols[2] if z_col else None

    if dimensions:
        plt.figure(figsize=dimensions)

    # Create Plot
    match type:
        case ChartType.SCATTER:
            sns.scatterplot(data, x=x_col, y=y_col, hue=z_col)

        case ChartType.LINE:
            sns.lineplot(data, x=x_col, y=y_col, hue=z_col)

        case ChartType.BAR:
            sns.barplot(data, x=x_col, y=y_col, hue=z_col)

    if title:
        plt.title(title)

    plt.xlabel(x_col)
    plt.ylabel(y_col)

    if plot_frontier:
        pass

    # Save or show plot
    if file_name:
        check_figures_dir()
        plt.savefig(FIGURES_DIR + file_name)
    else:
        plt.show()

    plt.clf()
