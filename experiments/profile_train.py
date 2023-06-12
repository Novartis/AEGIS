# -*- coding: utf-8 -*-
"""profiler_test.py

"""

import os
import pandas as pd


def main():
    epoch_0 = pd.read_csv(
        "outputs/2023-06-12/17-07-02/advanced_profiler/summary_0.csv"
    )
    epoch_3 = pd.read_csv(
        "outputs/2023-06-12/17-07-02/advanced_profiler/summary_3.csv"
    )

    epoch_0.sort_values(by=["cumulative_time"], ascending=False, inplace=True)
    epoch_3.sort_values(by=["cumulative_time"], ascending=False, inplace=True)
    epoch = pd.merge(
        epoch_0,
        epoch_3,
        on=["filename", "lineno", "function_name", "action_name"],
        how="inner",
    )
    epoch["diff_calls"] = (
        epoch["cumulative_time_y"] - epoch["cumulative_time_x"]
    )
    epoch["diff_n_calls"] = epoch["total_calls_y"] - epoch["total_calls_x"]

    epoch.sort_values(by=["diff_calls"], ascending=False, inplace=True)
    epoch.to_csv("experiments/profiler.csv")


if __name__ == "__main__":
    main()
