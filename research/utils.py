from tabulate import tabulate
import pandas as pd


def print_table(df: pd.DataFrame):

    print(
        "\n"
        + tabulate(
            df, headers="keys", tablefmt="heavy_grid", showindex=False, floatfmt=".2f"
        )
        + "\n"
    )
