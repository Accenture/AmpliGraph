import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
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
    Function that, given the degree of the subject and the object,
    returns the bin in which the triple should be placed.
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

def split_triples_by_degree_4way(triples, degrees, thresholds=[10, 30, 100]):
    """
    Split triples into 16 groups based on subject/object degrees using 4 degree levels:
    low, mid1, mid2, high.

    Parameters
    ----------
    triples : pd.DataFrame
        Must contain ["s", "p", "o"] columns.
    degrees : pd.Series or pd.DataFrame
        Node degrees. Index is node ID, values are degrees.
    thresholds : list of 3 numbers
        Degree thresholds to split into 4 groups: [low/mid1, mid1/mid2, mid2/high].

    Returns
    -------
    dict_splits : dict
        Keys are strings like "low-mid1", "high-low", etc.
        Values are indices of `triples` belonging to that group.
    """
    assert len(thresholds) == 3, "`thresholds` must contain exactly 3 values to form 4 degree groups."

    if isinstance(degrees, pd.DataFrame):
        degrees = degrees.squeeze()  # Convert to Series if needed

    t0, t1, t2 = thresholds
    low_nodes = degrees[degrees <= t0].index
    mid1_nodes = degrees[(degrees > t0) & (degrees <= t1)].index
    mid2_nodes = degrees[(degrees > t1) & (degrees <= t2)].index
    high_nodes = degrees[degrees > t2].index

    # print(f"Thresholds used: low ≤ {t0}, mid1 ({t0}, {t1}], mid2 ({t1}, {t2}], high > {t2}")
    # print(f"#low: {len(low_nodes)}, #mid1: {len(mid1_nodes)}, #mid2: {len(mid2_nodes)}, #high: {len(high_nodes)}")

    degree_groups = {
        "high": high_nodes,
        "mid2": mid2_nodes,
        "mid1": mid1_nodes,
        "low": low_nodes
    }

    dict_splits = dict()
    for subj_group, subj_nodes in degree_groups.items():
        for obj_group, obj_nodes in degree_groups.items():
            key = f"{subj_group}-{obj_group}"
            indices = triples[triples.s.isin(subj_nodes) & triples.o.isin(obj_nodes)].index
            dict_splits[key] = indices

    # # Summary print
    # print("Percentage of total triples per split:")
    # total = triples.shape[0]
    # for key, idxs in dict_splits.items():
    #     count = len(idxs)
    #     pct = round(100 * count / total, 2)
    #     print(f"{key.upper():<10}: {count} ({pct}%)")

    return dict_splits


def aggregated_to_tables(data, path):
    """
    This function generates:
      - one table with the aggregated metrics
      - two tables for the splits:
          * one comparing corr_s vs pre_corr_s
          * one comparing corr_o vs pre_corr_o
    It then saves them to CSV files.
    """
    rows_s = []
    rows_o = []

    for split, values in data.items():
        if split not in ['high-low', 'low-high']:
            continue

        # corr_s
        if "corr_s" in values and "pre_corr_s" in values:
            row_s = {"split": split}
            for m in values["corr_s"].keys():
                row_s[f"{m}"] = values["corr_s"][m]
            rows_s.append(row_s)

        # corr_o
        if "corr_o" in values and "pre_corr_o" in values:
            row_o = {"split": split}
            for m in values["corr_o"].keys():
                row_o[f"{m}"] = values["corr_o"][m]
            rows_o.append(row_o)

    df_corr_s = pd.DataFrame(rows_s).set_index("split").round(3)
    df_corr_o = pd.DataFrame(rows_o).set_index("split").round(3)

    # --- Save to files ---
    df_corr_s.to_csv(os.path.join(path, "split_metrics_corr_s.csv"))
    df_corr_o.to_csv(os.path.join(path, "split_metrics_corr_o.csv"))


######################## PLOTS
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
