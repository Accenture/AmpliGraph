# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Reporting for graph partition strategies.

This module provides reporting capabilities for partitioning strategies.
"""
import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm

from .datasets import load_fb15k_237
from .graph_partitioner import RandomVerticesGraphPartitioner

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PartitioningReporter:
    """Assesses the quality of partitioning according to chosen metrics and report it.

    Available metrics: edge cut, edge imbalance, vertex imbalance, time, memory usage.

    Parameters
    ----------
    partitionings:
            Data splits to be compared.

    Example
    -------

    >>>>quality = PartitioningReporter(partitionings)
    >>>>report = quality.report(visualize=False)
    """

    def __init__(self, partitionings):
        """Initialises PartitioningReporter.

        Parameters
        ----------
        partitionings:
             List of partitioning strategies.
        """
        self.partitionings = partitionings

    def get_edge_cut(self, k, partitions, avg_size=None):
        """Calculates mean edge cut across partitions in a single partitioning.

        Parameters
        ----------
        k: int
            Number of partitions.
        partitions:
            Partitions in one partitioning.

        Returns
        -------
        edge_cut: ndarray
            Average edge cut between partitions.
        """

        intersections = []
        logger.debug(partitions)
        for partition1 in partitions:
            logger.debug("Partition 1: {}".format(partition1))
            intersect = []
            for partition2 in partitions:
                if partition1 == partition2:
                    continue
                inter = partition1.intersect(partition2)
                logger.debug("Intersections: {}".format(inter))
                intersect.append(len(inter))
                logger.debug("Partition 2: {}".format(partition2))
            intersections.append(np.mean(intersect))
        logger.debug("Intersections: {}".format(intersections))
        edge_cut = np.mean(intersections)
        edge_cut_proportion = None
        if avg_size:
            # edge cut with respect to the average partition size
            edge_cut_proportion = (edge_cut * 100) / avg_size
        return edge_cut, edge_cut_proportion

    def get_edge_imbalance(self, avg_size, max_size):
        """Calculates edge imbalance of partitions.

        Parameters
        ----------
        avg_size: int
            Average size of partition.
        max_size: int
            Maximum size of partition.

        Returns
        -------
        edge_imb: float
            Edge imbalance
        """

        edge_imb = max_size / avg_size - 1
        return edge_imb

    def get_vertex_imbalance_and_count(self, partitions, vertex_count=False):
        """Calculates vertex imbalance of partitions, vertex count - counts number
           of vertices in each partition that estimates the size of partition.


        Parameters
        ----------
        partitions:
            Partitions in one partitioning.

        Returns
        -------
        vertex_imb: float
            Vertex imbalance.
        vertex_count: list
            List of counts, e.g., for 2 partitions the list will be of size two with vertex count for each
            partition: (5,6), for 3 partitions: (5,2,4).
        """
        lengths = []
        for partition in partitions:
            ents_len = partition.backend.mapper.get_entities_count()
            lengths.append(ents_len)

        vertex_imb = np.max(lengths) / np.mean(lengths) - 1
        if vertex_count:
            return vertex_imb, lengths
        else:
            return vertex_imb

    def get_average_deviation_from_ideal_size_vertices(self, partitions):
        """Metric that calculates the average difference between the
        ideal size of partition (in terms of vertices) and the real size.

        It is expressed as the percentage deviation from the ideal size.

        Parameters
        ----------
        partitions:
             Partitions in one partitioning.

        Returns
        -------
        percentage_dev: float
             Percentage vertex size partition deviation
        """
        k = len(partitions)
        sizes = []
        for partition in partitions:
            ents_len = partition.backend.mapper.get_entities_count()
            sizes.append(ents_len)
        data_size = ents_len
        ideal_size = data_size / k
        percentage_dev = (
            (np.sum([np.abs(ideal_size - size) for size in sizes]) / k)
            / ideal_size
        ) * 100
        return percentage_dev

    def get_average_deviation_from_ideal_size_edges(self, partitions):
        """Metric that calculates the average difference between the
        ideal size of partition (in terms of edges) and the real size.

        It is expressed as percentage deviation from ideal size.

        Parameters
        ----------
        partitions:
             Partitions in one partitioning.

        Returns
        -------
        percentage_dev: float
             Percentage edge size partition deviation.
        """

        k = len(partitions)
        sizes = []
        for partition in partitions:
            sizes.append(partition.get_data_size())
        logger.debug("Parent: {}".format(partition.parent.backend.data))
        data_size = partition.parent.get_data_size()
        logger.debug("Parent data size: {}".format(data_size))
        ideal_size = data_size / k
        logger.debug("Ideal data size: {}".format(ideal_size))
        percentage_dev = (
            (np.sum([np.abs(ideal_size - size) for size in sizes]) / k)
            / ideal_size
        ) * 100
        return percentage_dev

    def get_edges_count(self, partitions):
        """Counts number of edges in each partition that estimates the size of partition.

        Parameters
        ---------
        partitions:
             Partitions in one partitioning.

        Returns
        -------
        info: list
             List of counts, e.g. for 2 partitions the list will be of size two with the edge count: (10,12);
             for 3 partitions: (7,8,7).
        """
        info = []
        for partition in partitions:
            edges = partition.get_data_size()
            info.append(edges)

        return info

    def get_modularity(self):
        """Calculates modularity of partitions."""
        raise NotImplementedError

    def report_single_partitioning(
        self, partitioning, EDGE_CUT=True, EDGE_IMB=True, VERTEX_IMB=True
    ):
        """Calculate different metrics for a single partition.

        Parameters
        ----------
        partitioning:
            Single split of data into partitions.
        EDGE_CUT : bool
            Flag whether to calculate edge cut or not.
        EDGE_IMB : bool
            Flag whether to calculate edge imbalance or not.
        VERTEX_IMB : bool
            Flag whether to calculate vertex imbalance or not.

        Returns
        -------
        metrics: dict
            Dictionary with metrics.
        """
        logs = partitioning[1]
        partitioning = partitioning[0]
        tmp = partitioning.get_data()
        k = tmp.get_data_size()
        partitioner = partitioning
        partitioning = partitioner.get_partitions_list()
        sizes = [x.get_data_size() for x in partitioning]
        avg_size = np.mean(sizes)
        max_size = np.max(sizes)
        metrics = {"EDGE_IMB": None, "VERTEX_IMB": None, "EDGE_CUT": None}

        if logs:
            metrics["PARTITIONING TIME"] = logs["_SPLIT"]["time"]
            metrics["PARTITIONING MEMORY"] = logs["_SPLIT"]["memory-bytes"]
        if EDGE_CUT:
            edge_cut, edge_cut_proportion = self.get_edge_cut(
                k, partitioning, avg_size
            )
            metrics["EDGE_CUT"] = edge_cut
            metrics["EDGE_CUT_PERCENTAGE"] = edge_cut_proportion
        if EDGE_IMB:
            edge_imb = self.get_edge_imbalance(avg_size, max_size)
            metrics["EDGE_IMB"] = edge_imb
        if VERTEX_IMB:
            vertex_imb, vertex_count = self.get_vertex_imbalance_and_count(
                partitioning, vertex_count=True
            )
            metrics["VERTEX_IMB"] = vertex_imb
            metrics["VERTEX_COUNT"] = vertex_count
        metrics["EDGES_COUNT"] = self.get_edges_count(partitioning)
        metrics[
            "PERCENTAGE_DEV_EDGES"
        ] = self.get_average_deviation_from_ideal_size_edges(partitioning)
        metrics[
            "PERCENTAGE_DEV_VERTICES"
        ] = self.get_average_deviation_from_ideal_size_vertices(partitioning)
        partitioner.clean()
        return metrics

    def report(
        self, visualize=True, barh=True
    ):  # TODO: include plotting parameters
        """Collect individual reports for every partitioning.

        Parameters
        ----------
        visualize : bool
            Flag indicating whether to visualize output.

        Returns
        -------
        reports: dict
            Calculated metrics for all partitionings stored in a dictionary with keys the numbers of partitions
            and values the dictionary with metrics.
        """
        reports = {}
        for name, partitioning in self.partitionings.items():
            reports[name] = self.report_single_partitioning(
                partitioning, EDGE_IMB=True, VERTEX_IMB=True
            )
        k = len(self.partitionings[list(self.partitionings.keys())[0]][1])
        if visualize:
            plt.figure(
                figsize=(15, 15 + 0.3 * k + 0.1 * len(self.partitionings))
            )
            ind = 1
            row_size = 3
            size = int(len(reports[list(reports.keys())[0]]) / row_size) + 1
            for metric in reports[list(reports.keys())[0]]:
                plot = False
                dat = []
                color = iter(cm.PiYG(np.linspace(0, 1, len(reports))))
                colors_aggregate = {r: next(color) for r in reports}
                for j, report in enumerate(reports):
                    if reports[report][metric] is not None:
                        if isinstance(reports[report][metric], list):
                            n = len(reports[report][metric])
                            color = iter(cm.seismic(np.linspace(0, 1, n)))
                            colors = {
                                "partition {}".format(i): next(color)
                                for i in range(n)
                            }
                            width = 0.8 / n
                            for i, r in enumerate(reports[report][metric]):
                                label = "partition {}".format(i)
                                dat.append(
                                    {
                                        "y": j + (i * width),
                                        "width": r,
                                        "height": width,
                                        "label": label,
                                        "label2": str(report),
                                        "color": colors[label],
                                    }
                                )
                        else:
                            colors = colors_aggregate
                            label = str(report)
                            dat.append(
                                {
                                    "y": j,
                                    "width": reports[report][metric],
                                    "label2": label,
                                    "color": colors[label],
                                }
                            )
                        plot = True
                if plot:
                    plt.subplots_adjust(wspace=0.1, hspace=0.4)
                    plt.subplot(size, row_size, ind)

                    if barh:
                        unpacked = {k: [dic[k] for dic in dat] for k in dat[0]}
                        data = copy.deepcopy(unpacked)
                        del unpacked["label2"]
                        plt.barh(**unpacked, edgecolor="white")
                    else:
                        plt.bar(*list(zip(*dat)), edgecolor="white")

                    labels = list(colors.keys())
                    # handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
                    labels = []
                    for elem in data["label2"]:
                        if elem not in labels:
                            labels.append(elem)
                    # if type(labels) == set:
                    if (ind % row_size) == 1:
                        plt.yticks(range(len(list(labels))), list(labels))
                    else:
                        plt.yticks([])
                    plt.title(metric)
                    plt.xticks(rotation=70)
                    ind += 1
            plt.show()
        return reports


def compare_partitionings(
    list_of_partitioners, data, num_partitions=2, visualize=True
):
    """Wrapper around PartitioningReporter hiding logging settings.

    Parameters
    ---------
    list_of_partitioners: list
         List of uninitialized partitioners.
    data: ndarray
         Numpy array with the graph to be split into partitions.
    num_partitions: int
         Number of partitions required.
    visualize : bool
         Flag whether to visualize results or not.

    Returns
    -------
    result: dict
         Dictionary with metrics evaluating partitionings.

    Example
    -------
    >>>partitioners = [NaiveGraphPartitioner,
                       SortedEdgesGraphPartitioner,
                       DoubleSortedEdgesGraphPartitioner]
    >>>report = compare_partitionings(partitioners)
    """
    if isinstance(num_partitions, int):
        n_partitions = [num_partitions] * len(list_of_partitioners)
    else:
        n_partitions = num_partitions
    partitionings = {}
    for partitioner, n in zip(list_of_partitioners, n_partitions):
        logger.debug("Running: {}".format(partitioner.__name__))
        logs = {}
        if n != 0:
            data.reload()
        partitioner_fitted = partitioner(data, k=n, log=logs)
        partitionings[partitioner.__name__] = (partitioner_fitted, logs)
    reporter = PartitioningReporter(partitionings=partitionings)
    result = reporter.report(visualize=visualize, barh=True)
    return result


def main():
    """Main function with example usage."""
    from ampligraph.datasets import GraphDataLoader
    from ampligraph.datasets.sqlite_adapter import SQLiteAdapter

    sample = load_fb15k_237()["train"]
    data = GraphDataLoader(sample, backend=SQLiteAdapter, in_memory=False)
    partitioners = [RandomVerticesGraphPartitioner]
    report = compare_partitionings(partitioners, data, visualize=False)
    print(report)


#   Expected output:
#    {'RandomVerticesGraphPartitioner': {'EDGE_IMB': 0.40953499098494706,
#     'VERTEX_IMB': 0.03495702005730661,
#     'EDGE_CUT': 6736.0,
#     'PARTITIONING TIME': 139.55057835578918,
#     'PARTITIONING MEMORY': 7473904,
#     'EDGE_CUT_PERCENTAGE': 9.414790277719542,
#     'VERTEX_COUNT': [7224, 6736],
#     'EDGES_COUNT': [100848, 42246],
#     'PERCENTAGE_DEV_EDGES': 47.41414475497492,
#     'PERCENTAGE_DEV_VERTICES': 3.757325060324026}}

if __name__ == "__main__":
    main()
