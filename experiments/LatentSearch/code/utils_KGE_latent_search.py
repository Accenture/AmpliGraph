import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ampligraph.evaluation.metrics import mrr_score, hits_at_n_score
from ampligraph.datasets import load_fb15k_237, load_wn18rr, load_yago3_10


def load_dataset(dataset):
    """
    Function to load the datasets.
    """
    if dataset == "fb15k-237":
        X = load_fb15k_237()
    elif dataset == "wn18rr":
        X = load_wn18rr()
    elif dataset == "yago310":
        X = load_yago3_10()
    elif dataset == "biokg":
        X = dict()
        X["train"] = pd.read_csv(
            "/home/alberto/ampligraph_datasets/biokg/train.csv",
        ).values
        X["valid"] = pd.read_csv(
            "/home/alberto/ampligraph_datasets/biokg/valid_protein_disease_associations.csv"
        ).values
        X["test"] = pd.read_csv(
            "/home/alberto/ampligraph_datasets/biokg/test_protein_disease_associations.csv"
        ).values

    train = pd.DataFrame(X["train"], columns=['s', 'p', 'o'])
    valid = pd.DataFrame(X["valid"], columns=['s', 'p', 'o'])
    test = pd.DataFrame(X["test"], columns=['s', 'p', 'o'])

    return train, valid, test

def ranks_to_triples(ranks, group='sp'):
    """
    This function receives in input a dictionary 
    ranks[pair][epoch] = {"pre": ranks_pre, "new": ranks_new} and
    returns a dictionary where each pair is complemented with the missing
    element of the triple and each epoch has a dedicated column with the
    corresponding rank.
    """
    assert group == 'sp' or group == 'po'

    all_triples = []

    for pair, data in ranks.items():
        targets = data["targets"]
        base_cols = {
            "s": [pair[0]] * len(targets) if group == "sp" else targets,
            "p": [pair[1]] * len(targets),
            "o": targets if group == "sp" else [pair[0]] * len(targets),
            "pre": data["pre_ranks"],
        }

        # Add epoch columns efficiently
        epoch_cols = {
            f"epoch{i + 1}": data["new_ranks"][i]
            for i in range(len(data["new_ranks"]))
        }

        # Optional early stop column
        if "early_stop_new_ranks" in data:
            epoch_cols["early_stop_new_ranks"] = data["early_stop_new_ranks"]

        # Create DataFrame in a single call
        triples = pd.DataFrame({**base_cols, **epoch_cols})
        all_triples.append(triples)

    return pd.concat(all_triples, axis=0)


def map_degree_to_bin(subj_degree, obj_degree, thresholds=[10, 30, 100]):
    """
    Function that, given the degree of the subject and the object, returns the bin in which the triple should be placed.
    """
    def degree_to_bin(deg):
        if deg <= thresholds[0]:
            return "low"
        elif deg <= thresholds[1]:
            return "mid1"
        elif deg <= thresholds[2]:
            return "mid2"
        else:
            return "high"

    return f"{degree_to_bin(subj_degree)}-{degree_to_bin(obj_degree)}"

def group_by_pair(train, valid, test, group='sp'):
    """
    Function to group train, valid, and test based on (subject, predicate) or (predicate, object).
    """
    if group == 'sp':
        group_by = ['s', 'p']
    else:
        group_by = ['o', 'p']

    return train.groupby(group_by), valid.groupby(group_by), test.groupby(group_by)

def compute_metrics_across_epochs(
        all_triples_sp, all_triples_po, dict_splits=None,
        only_affected=False, col_splits="triple_type",
        aggregated=False
):
    """
    Compute the metrics across epoch. Either aggregate them across all test triples
    or compute them on different splits as per `dict_splits`.

    Parameters
    ----------
    only_affected: bool
        Whether to compute the metrics on the subset of triples for which
        the latent search could actually be performed (i.e., those triples
        for which we had examples in the training and validation set). In
        this way we can filter out those triples for which we have a lot of
        zero-difference ranks.
    """
    joined_ranks = all_triples_sp.merge(all_triples_po, how='inner', on=['s', 'p', 'o'])

    epochs = [el for el in joined_ranks.columns if el.startswith("epoch") and not el.endswith("_x")]
    if not only_affected:
        for i in range(1, len(epochs) + 1):
            mask_x = joined_ranks[f"epoch{i}_x"].isna()
            joined_ranks.loc[mask_x, f"epoch{i}_x"] = joined_ranks.loc[mask_x, "pre_x"]
            mask_y = joined_ranks[f"epoch{i}_y"].isna()
            joined_ranks.loc[mask_y, f"epoch{i}_y"] = joined_ranks.loc[mask_y, "pre_y"]
    else:
        mask_x = ~joined_ranks["epoch1_x"].isna()
        mask_y = ~joined_ranks["epoch1_y"].isna()

    metrics = dict()
    if not dict_splits:
        for i in range(len(epochs) + 1):
            target_cols = ['pre_x', 'pre_y'] if i == 0 else [f'epoch{i}_x', f'epoch{i}_y']
            metrics[i] = {"MRR": mrr_score(joined_ranks[target_cols].values)}
            for j in [1, 3, 10, 100]:
                metrics[i][f"Hits@{j}"] = hits_at_n_score(joined_ranks[target_cols].values, j)
    else:
        for split in dict_splits:
            metrics[split] = {"corr_o": dict(), "corr_s": dict()}
            split_mask = joined_ranks[f"{col_splits}_x"] == split
            for i in range(len(epochs) + 1):

                corrupt_obj = ["pre_x"] if i == 0 else [f"epoch{i}_x"]
                corrupt_subj = ["pre_y"] if i == 0 else [f"epoch{i}_y"]

                metrics[split]["corr_o"][i] = {
                    "MRR": mrr_score(
                        joined_ranks.loc[split_mask, corrupt_obj].values if not only_affected \
                            else joined_ranks.loc[split_mask & mask_x, corrupt_obj].values
                    )
                }
                metrics[split]["corr_s"][i] = {
                    "MRR": mrr_score(
                        joined_ranks.loc[split_mask, corrupt_subj].values if not only_affected \
                            else joined_ranks.loc[split_mask & mask_y, corrupt_subj].values
                    )
                }
                for j in [1, 3, 10, 100]:
                    metrics[split]["corr_o"][i][f"Hits@{j}"] = hits_at_n_score(
                        joined_ranks.loc[split_mask, corrupt_obj].values if not only_affected \
                            else joined_ranks.loc[split_mask & mask_x, corrupt_obj].values,
                        j
                    )
                    metrics[split]["corr_s"][i][f"Hits@{j}"] = hits_at_n_score(
                        joined_ranks.loc[split_mask, corrupt_subj].values if not only_affected \
                            else joined_ranks.loc[split_mask & mask_y, corrupt_subj].values,
                        j
                    )
    if aggregated:
        assert "early_stop_new_ranks" in all_triples_sp.columns and "early_stop_new_ranks" in all_triples_po.columns,\
            "You need to run the latent search with early stopping to get aggregated test metrics!"
        if not only_affected:
            mask_x = joined_ranks["early_stop_new_ranks_x"].isna()
            joined_ranks.loc[mask_x, "early_stop_new_ranks_x"] = joined_ranks.loc[mask_x, "pre_x"]
            mask_y = joined_ranks["early_stop_new_ranks_y"].isna()
            joined_ranks.loc[mask_y, "early_stop_new_ranks_y"] = joined_ranks.loc[mask_y, "pre_y"]
        else:
            mask_x = ~joined_ranks["early_stop_new_ranks_x"].isna()
            mask_y = ~joined_ranks["early_stop_new_ranks_y"].isna()

        aggregated_metrics = dict()
        target_cols = ["early_stop_new_ranks_x", "early_stop_new_ranks_y"]
        pre_target_cols = ["pre_x", "pre_y"]
        aggregated_metrics["aggregated"] = {"MRR": mrr_score(joined_ranks[target_cols].values)}
        aggregated_metrics["aggregated_pre"] = {"MRR": mrr_score(joined_ranks[pre_target_cols].values)}
        for j in [1, 3, 10, 100]:
            aggregated_metrics["aggregated"][f"Hits@{j}"] = hits_at_n_score(joined_ranks[target_cols].values, j)
            aggregated_metrics["aggregated_pre"][f"Hits@{j}"] = hits_at_n_score(joined_ranks[pre_target_cols].values, j)

        for split in dict_splits:
            aggregated_metrics[split] = {"corr_o": dict(), "corr_s": dict(), "pre_corr_o": dict(), "pre_corr_s": dict()}
            split_mask = joined_ranks[f"{col_splits}_x"] == split

            aggregated_metrics[split]["corr_o"] = {
                "MRR": mrr_score(
                    joined_ranks.loc[split_mask, "early_stop_new_ranks_x"].values if not only_affected \
                        else joined_ranks.loc[split_mask & mask_x, "early_stop_new_ranks_x"].values
                )
            }
            aggregated_metrics[split]["corr_s"] = {
                "MRR": mrr_score(
                    joined_ranks.loc[split_mask, "early_stop_new_ranks_y"].values if not only_affected \
                        else joined_ranks.loc[split_mask & mask_y, "early_stop_new_ranks_y"].values
                )
            }
            aggregated_metrics[split]["pre_corr_o"] = {
                "MRR": mrr_score(
                    joined_ranks.loc[split_mask, "pre_x"].values if not only_affected \
                        else joined_ranks.loc[split_mask & mask_x, "pre_x"].values
                )
            }
            aggregated_metrics[split]["pre_corr_s"] = {
                "MRR": mrr_score(
                    joined_ranks.loc[split_mask, "pre_y"].values if not only_affected \
                        else joined_ranks.loc[split_mask & mask_y, "pre_y"].values
                )
            }
            for j in [1, 3, 10, 100]:
                aggregated_metrics[split]["corr_o"][f"Hits@{j}"] = hits_at_n_score(
                    joined_ranks.loc[split_mask, "early_stop_new_ranks_x"].values if not only_affected \
                        else joined_ranks.loc[split_mask & mask_x, "early_stop_new_ranks_x"].values,
                    j
                )
                aggregated_metrics[split]["corr_s"][f"Hits@{j}"] = hits_at_n_score(
                    joined_ranks.loc[split_mask, "early_stop_new_ranks_y"].values if not only_affected \
                        else joined_ranks.loc[split_mask & mask_y, "early_stop_new_ranks_y"].values,
                    j
                )
                aggregated_metrics[split]["pre_corr_o"][f"Hits@{j}"] = hits_at_n_score(
                    joined_ranks.loc[split_mask, "pre_x"].values if not only_affected \
                        else joined_ranks.loc[split_mask & mask_x, "pre_x"].values,
                    j
                )
                aggregated_metrics[split]["pre_corr_s"][f"Hits@{j}"] = hits_at_n_score(
                    joined_ranks.loc[split_mask, "pre_y"].values if not only_affected \
                        else joined_ranks.loc[split_mask & mask_y, "pre_y"].values,
                    j
                )

        return metrics, aggregated_metrics

    return metrics


def aggregated_to_tables(data, path):
    """
    This function generates:
      - one table with the aggregated metrics
      - two tables for the splits:
          * one comparing corr_s vs pre_corr_s
          * one comparing corr_o vs pre_corr_o
    It then saves them to CSV files.
    """
    # --- Table 1: Aggregated ---
    agg_rows = []
    for k, v in data.items():
        if k.startswith("aggregated"):
            row = {"split": k}
            row.update(v)
            agg_rows.append(row)

    df_aggregated = pd.DataFrame(agg_rows)
    df_aggregated = df_aggregated.round(3)
    df_aggregated = df_aggregated.set_index("split")

    # --- Table 2a: corr_s vs pre_corr_s ---
    rows_s = []
    # --- Table 2b: corr_o vs pre_corr_o ---
    rows_o = []

    for split, values in data.items():
        if split.startswith("aggregated"):
            continue

        # corr_s
        if "corr_s" in values and "pre_corr_s" in values:
            row_s = {"split": split}
            for m in values["corr_s"].keys():
                row_s[f"{m}"] = values["corr_s"][m]
                row_s[f"{m} (pre)"] = values["pre_corr_s"][m]
            rows_s.append(row_s)

        # corr_o
        if "corr_o" in values and "pre_corr_o" in values:
            row_o = {"split": split}
            for m in values["corr_o"].keys():
                row_o[f"{m}"] = values["corr_o"][m]
                row_o[f"{m} (pre)"] = values["pre_corr_o"][m]
            rows_o.append(row_o)

    df_corr_s = pd.DataFrame(rows_s).set_index("split").round(3)
    df_corr_o = pd.DataFrame(rows_o).set_index("split").round(3)

    # --- Save to files ---
    df_aggregated.to_csv(os.path.join(path, "aggregated_metrics.csv"))
    df_corr_s.to_csv(os.path.join(path, "split_metrics_corr_s.csv"))
    df_corr_o.to_csv(os.path.join(path, "split_metrics_corr_o.csv"))

    # # --- Convert to Markdown ---
    # markdown_aggregated = "## Aggregated Metrics\n" + df_aggregated.to_markdown() + "\n"
    # markdown_splits = "## Split Metrics\n" + df_splits.to_markdown() + "\n"
    #
    # # --- Save to files ---
    # with open(os.path.join(path, "results_aggregated.md"), "w") as f:
    #     f.write(markdown_aggregated)
    #     f.write("\n------------------------------------------------\n\n")
    #     f.write(markdown_splits)





######################## PLOTS
def plot_ranks_improvement_across_epochs(
        all_triples,
        splits,
        group='sp',
        path="/users/alberto.bernardi/AmpliGraph-Lab/experiments/LatentSearch/results/KGE_inference_search/v0.1/",
        savefig=True,
        four_way=False,
        colored=False,
):
    """
    This function generates a PDF with as many pages as the number of epochs
    for which we want to plot the histograms with the distribution of the
    difference between the original rank (before the latent search) and the
    new rank (after the latent search).
    """
    num_epochs = len([col for col in all_triples.columns if col.startswith("epoch")])

    file_name = f"ranks_improvement_across_epochs_{group}.pdf"
    if colored:
        file_name = 'col_' + file_name
    if four_way:
        file_name = '4way_' + file_name
    if not savefig:
        file_name = 'tmp_' + file_name

    file_path = os.path.join(path, file_name)
    with PdfPages(file_path) as pdf:
        for epoch in range(1, num_epochs + 1):
            if not four_way:
                fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            else:
                fig, axes = plt.subplots(4, 4, figsize=(20, 20))
            axes = axes.flatten()

            bins = np.array([-10 ** 4, -1000, -100, -10, -0.9, 0.9, 10, 100, 1000, 10 ** 4])

            for idx, split in enumerate(splits):
                subset = all_triples[all_triples.triple_type == split]
                diff = subset["pre"] - subset[f"epoch{epoch}"]

                # Histogram manually to avoid bin crossing zero
                counts, edges = np.histogram(diff, bins=bins)
                centers = 0.5 * (edges[1:] + edges[:-1])

                axes[idx].bar(centers, counts, width=np.diff(edges), color="skyblue", edgecolor="k")
                axes[idx].set_title(f"{split} - {int((all_triples.triple_type == split).sum())} Triples")
                axes[idx].set_xlabel("Rank Improvement")
                axes[idx].set_ylabel("Count")
                axes[idx].set_xscale('symlog')
                axes[idx].set_yscale('log')

                axes[idx].axvline(0, color='b', linestyle='--', linewidth=1)
                # Count improvements
                count_le_0 = (diff < 0).sum()
                count_gt_0 = (diff > 0).sum()

                # Annotate counts to left and right of line
                axes[idx].text(0.01, 0.98, f"{count_le_0}", color='b', ha='left', va='top',
                               transform=axes[idx].transAxes, fontsize=10)
                axes[idx].text(0.99, 0.98, f"{count_gt_0}", color='b', ha='right', va='top',
                               transform=axes[idx].transAxes, fontsize=10)

                # Optional: Add count labels on top of bars
                for c, x in zip(counts, centers):
                    if c > 0:
                        axes[idx].text(x, c, str(c), ha='center', va='bottom', fontsize=8)

                if colored:
                    if count_le_0 < count_gt_0:
                        axes[idx].patch.set_facecolor('green')
                        axes[idx].patch.set_alpha(count_gt_0 / (count_le_0 + count_gt_0 + 1e-5))
                    else:
                        axes[idx].patch.set_facecolor('red')
                        axes[idx].patch.set_alpha(count_le_0 / (count_le_0 + count_gt_0 + 1e-5))

            plt.suptitle(f"Histogram of Rank Improvements - Before vs Epoch {epoch}", y=0.98, fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            if savefig:
                pdf.savefig(fig)
            else:
                plt.show()
            plt.close(fig)



def plot_metrics_across_epochs(
        metrics, dict_splits=None, display_metric=None,
        savefig=False, path=None, only_affected=False, valid=False, flat=False
):
    """
    Plot the metrics across epochs. Either aggregate them across all test triples
    or plot them based on dict splits.
    """
    if not dict_splits:
        df_metrics = pd.DataFrame(metrics)
        plt.figure(figsize=(20, 10))
        colors = ["blue", "green", "pink", "orange", "red"]
        metrics = df_metrics.index

        for i, metric in enumerate(metrics):
            y = df_metrics.loc[metric]
            plt.plot(df_metrics.columns, y, label=metric, marker='o', color=colors[i])

            for j, val in enumerate(y):
                col = df_metrics.columns[j]
                rounded_val = round(val, 3)

                # Label the first point and the max point only
                if j == 0 or val == y.max():
                    plt.text(col, val + 0.01, f"{rounded_val}", ha='center', fontsize=10, color=colors[i])

    else:
        if flat:
            fig, axes = plt.subplots(len(dict_splits), figsize=(20, 10), sharey=True)
        else:
            fig, axes = plt.subplots(
                int(len(dict_splits) ** 0.5), int(len(dict_splits) ** 0.5),
                figsize=(20, 20), sharex=True, sharey=True
            )
        axes = axes.flatten()

        for split_idx, split in enumerate(dict_splits):
            ax = axes[split_idx]

            curr_metrics = metrics[split]
            corr_s = pd.DataFrame(curr_metrics["corr_s"])
            corr_o = pd.DataFrame(curr_metrics["corr_o"])

            if display_metric is None:
                display_metric = ["MRR", "Hits@1", "Hits@3", "Hits@10", "Hits@100"]
            for metric in display_metric:
                y_s = corr_s.loc[metric]
                y_o = corr_o.loc[metric]

                ax.plot(corr_s.columns, y_s, label=f"{metric} - Subj. Corr.", marker='o')
                ax.plot(corr_o.columns, y_o, label=f"{metric} - Obj. Corr.", marker='o')

                flag_s = True
                flag_o = True
                for j, col in enumerate(corr_s.columns):
                    # Subject annotations
                    val_s = round(y_s[col], 3)
                    val_o = round(y_o[col], 3)
                    if j == 0:
                        ax.text(col, y_s[col] + 0.02, f"{val_s}", ha='center', fontsize=8)
                        ax.text(col, y_o[col] + 0.02, f"{val_o}", ha='center', fontsize=8)

                    if y_s[col] == y_s.max() and flag_s:
                        ax.text(col, y_s[col] + 0.02, f"{val_s}", ha='center', fontsize=8)
                        flag_s = False

                    if y_o[col] == y_o.max() and flag_o:
                        ax.text(col, y_o[col] + 0.02, f"{val_o}", ha='center', fontsize=8)
                        flag_o = False

            ax.legend(fontsize=8)
            ax.set_title(split)
            ax.grid(True)

    if not dict_splits:
        plt.title(f"Evaluation Metrics Over Epochs{' - Validation Set' if valid else ''}", fontsize=20)
        plt.legend(loc='upper right')
        plt.grid(True)
    else:
        plt.suptitle(f"Evaluation Metrics Over Epochs{' - Validation Set' if valid else ''}", fontsize=20, y=1.0)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True)
    plt.tight_layout()
    if savefig:
        assert path is not None, "Provide a path where to save the plot!"
        name_file = f"{'valid_' if valid else ''}{'only_affected_' if only_affected else ''}metrics_{'aggregated' if not dict_splits else 'by_split'}.png"
        plt.savefig(os.path.join(path, name_file))
        plt.close()
    else:
        plt.show()

def plot_metrics_two_splits(
    metrics,
    display_metric=None,
    savefig=False,
    path=None,
    valid=False
):
    """
    Plot only 'high-low' and 'low-high' splits from the metrics dict.
    """

    REQUIRED_SPLITS = ["high-low", "low-high"]

    # Check required splits exist
    for split in REQUIRED_SPLITS:
        if split not in metrics:
            raise ValueError(f"Split '{split}' not found in metrics.")

    # Default metrics
    if display_metric is None:
        display_metric = ["MRR", "Hits@1", "Hits@3", "Hits@10", "Hits@100"]

    # Create two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, split in zip(axes, REQUIRED_SPLITS):

        curr_metrics = metrics[split]
        corr_s = pd.DataFrame(curr_metrics["corr_s"])
        corr_o = pd.DataFrame(curr_metrics["corr_o"])

        for metric in display_metric:

            if split == "low-high":
                y_s = corr_s.loc[metric]
                ax.plot(corr_s.columns, y_s, label=f"{metric} - Subj. Corr.", linewidth=2)
            elif split == "high-low":
                y_o = corr_o.loc[metric]
                ax.plot(corr_o.columns, y_o, label=f"{metric} - Obj. Corr.", linewidth=2)

            ax.tick_params(axis='both', which='major', labelsize=16)

        ax.set_title(f"{split}", fontsize=20)
        ax.grid(True)
        ax.set_xlabel("Epoch", fontsize=18)
        ax.set_xlim((0, 100))
        ax.set_ylim((0, 0.5))
        ax.legend(fontsize=16)

    fig.suptitle(
        f"Evaluation Metrics Over Epochs{' - Validation Set' if valid else ''}",
        fontsize=24, fontweight="bold"
    )
    plt.tight_layout()

    if savefig:
        assert path is not None, "Provide a path where to save the plot!"
        filename = f"{'valid_' if valid else ''}metrics_highlow_lowhigh.png"
        plt.savefig(os.path.join(path, filename))
        plt.close()
    else:
        plt.show()

def barplot_diff_rank_over_epochs_by_split(all_triples, dict_splits, group='sp', savefig=False, path=None):
    """
    This function generates a plot with one subplot for every split and
    it groups the rank difference into barplots that are clustered by epoch,
    so that you can right away observe how the bars for each split and rank
    difference values evolve across the latent search epochs.
    """

    epochs = [el for el in all_triples if el.startswith("epoch")]
    num_epochs = len(epochs)

    # Define bins (same for all epochs)
    bins = np.array([-10 ** 4, -10 ** 2, -10, -0.9, 0.9, 10, 100, 10 ** 4])
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    num_bins = len(bin_centers)

    bar_width = 0.9
    x_base = np.arange(num_bins) * (num_epochs + 1)

    fig, axes = plt.subplots(len(dict_splits), 1, figsize=(20, 3 * len(dict_splits)), sharex=True, sharey=True)

    if len(dict_splits) == 1:
        axes = [axes]  # make iterable if only 1 subplot

    for split_idx, split in enumerate(dict_splits):
        ax = axes[split_idx]

        for epoch_idx, epoch in enumerate(epochs):
            subset = all_triples[all_triples.triple_type == split]
            diff = subset["pre"] - subset[f"{epoch}"]
            counts, _ = np.histogram(diff, bins=bins)

            x_shifted = x_base + epoch_idx * bar_width
            ax.bar(x_shifted, counts, width=bar_width, label=f"Epoch {epoch_idx + 1}")

        ax.set_ylabel(f"{split}")
        ax.set_yscale('log')
        ax.set_xticks(x_base + (num_epochs / 2) * bar_width)
        ax.set_xticklabels([f"[{int(bins[i])}, {int(bins[i + 1])}]" for i in range(num_bins)],
                           rotation=45, ha='right')

    axes[-1].set_xlabel("Rank Improvement Bins")
    corruption = "Subject" if group == 'po' else "Object"
    plt.suptitle(f"Histogram of Rank Improvements Across Epochs - {corruption} Corruption", fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if savefig:
        plt.savefig(os.path.join(path, f"{group}_evolution_over_epochs.png"))
        plt.close()
    else:
        plt.show()


def make_rank_sankey(all_ranks, dict_splits, savefig=False, group='sp', path=None):
    epochs = [col for col in all_ranks.columns if col.startswith('epoch')]
    all_ranks.loc[all_ranks.epoch1.isna(), epochs] = all_ranks.loc[all_ranks.epoch1.isna(), "pre"]
    best_epochs = dict()
    for split_name in dict_splits:
        subset_all_ranks = all_ranks.loc[all_ranks.triple_type == split_name, epochs]
        best_epochs[split_name] = subset_all_ranks.apply(
            lambda x: hits_at_n_score(x.values, 10), axis=0
        ).idxmax()

    # all_ranks.loc[all_ranks.epoch1.isna(), "early_stop_new_ranks"] = all_ranks.loc[all_ranks.epoch1.isna(), "pre"]
    # for split_name in dict_splits:
    #     subset_all_ranks = all_ranks.loc[all_ranks.triple_type == split_name, "early_stop_new_ranks"]
    #     best_epochs[split_name] = subset_all_ranks.apply(
    #         lambda x: hits_at_n_score(x.values, 10), axis=0
    #     ).idxmax()

    # Define intervals
    intervals = [
        (1, 3), (4, 10), (11, 50),
        (51, 100), (101, 500), (501, np.inf)
    ]
    interval_labels = [
        '[1-3]', '[4-10]', '[11-50]',
        '[51-100]', '[101-500]', '>500'
    ]
    colors = [
        '#0072B2', '#E69F00', '#009E73', '#D55E00', 'lightgreen', 'lightskyblue'
    ]
    color_map = {label: color for label, color in zip(interval_labels, colors)}

    # Helper to map a rank to an interval label
    def map_to_interval(val):
        for (start, end), label in zip(intervals, interval_labels):
            if start <= val <= end:
                return label
        return 'out-of-range'

    num_rows = int(np.sqrt(len(dict_splits)))
    fig = make_subplots(
        rows=num_rows, cols=num_rows, specs=[[{'type': 'sankey'}] * num_rows] * num_rows,
        subplot_titles=[f"{split_name} - {int((all_ranks.triple_type == split_name).sum())} Triples"
                        for split_name in dict_splits],
        vertical_spacing=0.05
    )

    for i, split_name in enumerate(dict_splits):
        # Defining the mask to select the values of interest
        array1 = all_ranks.loc[all_ranks.triple_type == split_name, "pre"]
        array2 = all_ranks.loc[all_ranks.triple_type == split_name, best_epochs[split_name]]

        # Map both arrays to interval labels
        group1 = np.array([map_to_interval(x) for x in array1])
        group2 = np.array([map_to_interval(x) for x in array2])

        # Count transitions between interval labels
        df = pd.DataFrame({'from': group1, 'to': group2})
        flow_counts = df.value_counts().reset_index(name='count')

        not_represented_labels = set(interval_labels).difference(flow_counts['from'].unique())
        for label in not_represented_labels:
            if label in not_represented_labels:
                flow_counts = pd.concat([flow_counts, pd.DataFrame({'from': [label], 'to': [label], 'count': [0]})])

        # Create mapping from labels to node indices
        label_index = {label: i for i, label in enumerate(interval_labels)}

        # Create Sankey diagram components
        source = [label_index[f] for f in flow_counts['from']]
        target = [label_index[t] + len(interval_labels) for t in flow_counts['to']]
        value = flow_counts['count'].tolist()
        link_colors = [color_map[f] for f in flow_counts['from']]

        # Final labels: from and to
        node_labels = [f'Before: {lbl}' for lbl in interval_labels] + [f'After: {lbl}' for lbl in interval_labels]
        node_colors = [color_map[lbl] for lbl in interval_labels] + [color_map[lbl] for lbl in interval_labels]

        # Sankey diagram
        subfig = go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20,
                thickness=30,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors # optional: match node color to group
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors
            )
        )
        fig.add_trace(subfig, row=i // num_rows + 1, col=i % num_rows + 1)

    fig.update_layout(
        title_text=f"Rank Transition - {'Subject' if group == 'po' else 'Object'} Corruption",
        font_size=12, width=2000, height=3600
    )
    if savefig:
        fig.write_html(os.path.join(path, f"sankey_ranks_{group}.html"), full_html=True, include_plotlyjs='cdn')
    else:
        fig.show()


def new_make_rank_sankey(all_ranks, dict_splits, savefig=False, group='sp', path=None):
    # Fill NaNs in early_stop_new_ranks with pre values
    all_ranks["early_stop_new_ranks"] = all_ranks["early_stop_new_ranks"].fillna(all_ranks["pre"])

    # Define intervals
    intervals = [
        (1, 3), (4, 10), (11, 50),
        (51, 100), (101, 500), (501, np.inf)
    ]
    interval_labels = [
        '[1-3]', '[4-10]', '[11-50]',
        '[51-100]', '[101-500]', '>500'
    ]
    colors = [
        '#0072B2', '#E69F00', '#009E73', '#D55E00', 'lightgreen', 'lightskyblue'
    ]
    color_map = {label: color for label, color in zip(interval_labels, colors)}

    # Helper to map a rank to an interval label
    def map_to_interval(val):
        for (start, end), label in zip(intervals, interval_labels):
            if start <= val <= end:
                return label
        return 'out-of-range'

    num_rows = int(np.sqrt(len(dict_splits)))
    fig = make_subplots(
        rows=num_rows, cols=num_rows, specs=[[{'type': 'sankey'}] * num_rows] * num_rows,
        subplot_titles=[f"{split_name.upper()} - {int((all_ranks.triple_type == split_name).sum())} Triples"
                        for split_name in dict_splits],
        vertical_spacing=0.05
    )

    for i, split_name in enumerate(dict_splits):
        # Defining the mask to select the values of interest
        array1 = all_ranks.loc[all_ranks.triple_type == split_name, "pre"]
        array2 = all_ranks.loc[all_ranks.triple_type == split_name, "early_stop_new_ranks"]

        # Map both arrays to interval labels
        group1 = np.array([map_to_interval(x) for x in array1])
        group2 = np.array([map_to_interval(x) for x in array2])

        # Count transitions between interval labels
        df = pd.DataFrame({'from': group1, 'to': group2})
        flow_counts = df.value_counts().reset_index(name='count')

        not_represented_labels = set(interval_labels).difference(flow_counts['from'].unique())
        for label in not_represented_labels:
            flow_counts = pd.concat([flow_counts, pd.DataFrame({'from': [label], 'to': [label], 'count': [0]})])

        # Create mapping from labels to node indices
        label_index = {label: i for i, label in enumerate(interval_labels)}

        # Create Sankey diagram components
        source = [label_index[f] for f in flow_counts['from']]
        target = [label_index[t] + len(interval_labels) for t in flow_counts['to']]
        value = flow_counts['count'].tolist()
        link_colors = [color_map[f] for f in flow_counts['from']]

        # Final labels: from and to
        node_labels = [f'Before: {lbl}' for lbl in interval_labels] + [f'After: {lbl}' for lbl in interval_labels]
        node_colors = [color_map[lbl] for lbl in interval_labels] + [color_map[lbl] for lbl in interval_labels]

        # Sankey diagram
        subfig = go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20,
                thickness=30,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors,
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors
            )
        )
        fig.add_trace(subfig, row=i // num_rows + 1, col=i % num_rows + 1)

    fig.update_layout(
        title_text=f"Rank Transition - {'Subject' if group == 'po' else 'Object'} Corruption",
        font_size=18, width=2000, height=3600
    )
    if savefig:
        fig.write_html(os.path.join(path, f"new_sankey_ranks_{group}.html"), full_html=True, include_plotlyjs='cdn')
    else:
        fig.show()
