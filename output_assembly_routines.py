#!/usr/bin/env python3

has_pandas = True
try:
    import pandas
except ModuleNotFoundError:
    has_pandas = False

import os
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter


outdir = os.path.dirname(os.path.realpath(__file__))  # this file's directory


def get_empty_figure(grid=False, subplots=(1, 1), **kwargs):
    n, m = subplots
    kwargs.setdefault("figsize", (m * 5.0, n * 5.0 / 1.618))
    kwargs.setdefault("dpi", 150)

    fig = plt.figure(**kwargs)
    gs = fig.add_gridspec(n, m)
    axes = [fig.add_subplot(gs[i, j]) for j in range(m) for i in range(n)]
    for ax in axes:
        ax.tick_params(which="both", direction="in", right=True, top=True)
        if grid:
            ax.grid(axis="both", which="both", linestyle=":", linewidth=0.8)
    return fig, axes


def generate_output(results_file):
    print(f"Generating output from {results_file}...")
    flushed_output = []

    data = pandas.read_csv(results_file)
    data = data.sort_values("num_procs")

    fig = plt.figure(figsize=(5.0, 5.0 / 1.618), dpi=150)
    ax = fig.gca()
    ax.set_xlabel(r"# MPI processes")
    ax.set_ylabel(r"time [s]")
    ax.set_xscale("log", basex=2)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    plot_opts = dict(markerfacecolor="none", linewidth=0.8)

    for assembler, marker, color in [
        ("mono", "o", "C0"), ("block", "s", "C1"), ("nest", "v", "C2"), ("split", "d", "C3")
    ]:
        subdata = data.loc[data["assembler"] == assembler]
        if not subdata.empty:
            print("")
            print(subdata.iloc[:, 0:12])
            print("")
            print(subdata.iloc[:, 12:16])
            print("")
            print(subdata.iloc[:, 16:])
            print("")
            num_procs = subdata["num_procs"].to_numpy()
            plot_opts["linestyle"] = "-"
            ax.plot(num_procs, subdata["t_mat"].to_numpy(), label=f"{assembler} (Mat)", color=color, marker=marker, **plot_opts)
            plot_opts["linestyle"] = "--"
            ax.plot(num_procs, subdata["t_vec"].to_numpy(), label=f"{assembler} (Vec)", color=color, marker=marker, **plot_opts)

    ax.legend(loc=0)
    fig.tight_layout()

    filename = os.path.join(outdir, f"{os.path.splitext(results_file)[0]}.png")
    fig.savefig(filename)
    flushed_output.append(filename)

    print("Generated output:")
    for filename in flushed_output:
        print(f"  + {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script generating output for `test_assembly_routines.py`;"
                    " requires `pandas` library!"
    )
    parser.add_argument("results_file", metavar="FILENAME", type=str, help="CSV file with results")
    args = parser.parse_args()

    if has_pandas:
        generate_output(args.results_file)
    else:
        msg = """This script requires pandas, install it with
        python3 -m pip install pandas"""
        raise RuntimeError(msg)
