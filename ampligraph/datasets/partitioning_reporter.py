import numpy as np
import matplotlib.pyplot as plt
import ampligraph as amp
from ampligraph import datasets
import tracemalloc
from time import time
import pytest
from abc import ABC, abstractmethod
from functools import wraps


def get_memory_size():
    """Get memory size.

    Returns
    -------
    total: memory size in usage in total
    """
    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics('lineno', cumulative=True)
    total = sum(stat.size for stat in stats)
    return total

def get_human_readable_size(size_in_bytes):
    """Convert size in bytes in human readable units.

    Parameters
    ----------
    size_in_bytes: original size given in bytes

    Returns
    -------
    tuple of new size and unit, size in units GB/MB/KB/Bytes according to thresholds.
    """
    if size_in_bytes >= 1024*1024*1024:
        return float(size_in_bytes/(1024*1024*1024)), "GB" # return in GB
    if size_in_bytes >= 1024*1024:
        return float(size_in_bytes/(1024*1024)), "MB" # return in MB
    if size_in_bytes >= 1024:
        return float(size_in_bytes/1024), "KB" # return in KB
    return float(size_in_bytes), "Bytes"

def timing_and_memory(f):
    """Decorator to register time and memory used by a function f.

    Parameters
    ----------
    f: function for which the time an memory will be measured

    It logs the time and the memory in the dictionary passed inside 'log' parameter if provided.
    Time is logged in seconds, memory in bytes. Example dictionary entry looks like that:
    {'SPLIT':{'time':1.62, 'memory-bytes':789.097}}, where keys are names of functions that
    were called to get the time measured in uppercase.

    Requires
    --------
    passing **kwargs in function parameters
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        mem_before = get_memory_size()
        start = time()
        result = f(*args, **kwargs)
        end = time()
        mem_after = get_memory_size()
        mem_diff = mem_after - mem_before
        print("{}: memory before: {:.5}{}, after: {:.5}{}, consumed: {:.5}{}; exec time: {:.5}s".format(
            f.__name__,
            *get_human_readable_size(mem_before), *get_human_readable_size(mem_after), *get_human_readable_size(mem_diff), end-start))

        if 'log' in kwargs:
            name = kwargs.get('log_name', f.__name__.upper())
            kwargs['log'][name] = {'time': end - start, 'memory-bytes': mem_diff}
        return result
    return wrapper


class PartitioningReporter:
    """Assesses the quality of partitioning according to
    chosen metrics and report it.

    Available metrics: edge cut, edge imbalance, vertex imbalance, time, memory usage.

    Parameters
    ----------
    partitionings: data splits to be compared

    Example
    -------

    >>>>quality = PartitioningReporter(partitionings)
    >>>>report = quality.report(visualize=False)
    """
    def __init__(self, partitionings):
        self.partitionings = partitionings

    def get_edge_cut(self, k, partitions, avg_size=None):
        """Calculates mean edge cut across partitions in a single
        partitioning.

        Parameters
        ----------
        k: number of partitions
        partitions: partitions in one partitioning

        Returns
        -------

        edge_cut: average edge cut between partitions"""

        intersections = []
        for i in range(k):
            tmp = []
            for j in range(k):
                tmp.append(len(np.intersect1d(partitions[i], partitions[j])))
            intersections.append(np.mean(tmp))

        edge_cut = np.mean(intersections)
        edge_cut_proportion = None
        if avg_size:
            edge_cut_proportion = (edge_cut*100)/avg_size # edge cut with respect to the average partition size
        return edge_cut, edge_cut_proportion

    def get_edge_imbalance(self, avg_size, max_size):
        """Calculates edge imbalance of partitions

        Parameters
        ----------
        avg_size: average size of partition
        max_size: maximum size of partition

        Returns
        -------
        edge_imb: edge imbalance
        """

        edge_imb = max_size/avg_size - 1
        return edge_imb

    def get_vertex_imbalance(self, partitions):
        """Calculates vertex imbalance of partitions

        Parameters
        ----------
        partitions: partitions in one partitioning

        Returns
        -------
        vertex_imb: vertex imbalance
        """
        lengths = []
        for partition in partitions:
            lengths.append(len(np.asarray(list(set(partition[:,0]).union(set(partition[:,2]))))))

        vertex_imb = np.max(lengths)/np.mean(lengths) - 1
        return vertex_imb

    def get_modularity(self):
        """Calculates modularity of partitions.

        Parameters
        ----------

        Returns
        -------
        modularity: modularity
        """
        raise NotImplementedError

    def report_single_partitioning(self, partitioning, EDGE_CUT=True, \
                                   EDGE_IMB=True, VERTEX_IMB=True):
        """Calculate different metrics for a single partition.

        Parameters
        ----------
        partitioning: single split of data into partitions
        EDGE_CUT [True/False]: flag whether to calculate edge cut or not
        EDGE_IMB [True/False]: flag whether to calculate edge imbalance or not
        VERTEX_IMB [True/False]: flag whether to calculate vertex imbalance or not

        Returns
        -------
        metrics: dictionary with metrics
        """
        logs = partitioning[1]
        partitioning = partitioning[0]
        k = len(partitioning)
        sizes = [len(x) for x in partitioning]
        avg_size = np.mean(sizes)
        max_size = np.max(sizes)
        metrics = {"EDGE_IMB": None, "VERTEX_IMB": None, "EDGE_CUT": None}

        if logs:
            metrics["PARTITIONING TIME"] = logs["SPLIT"]['time']
            metrics["PARTITIONING MEMORY"] = logs["SPLIT"]['memory-bytes']
        if EDGE_CUT:
            edge_cut, edge_cut_proportion = self.get_edge_cut(k, partitioning, avg_size)
            metrics["EDGE_CUT"] = edge_cut
            metrics["EDGE_CUT_PERCENTAGE"] = edge_cut_proportion
        if EDGE_IMB:
            edge_imb = self.get_edge_imbalance(avg_size, max_size)
            metrics["EDGE_IMB"] = edge_imb
        if VERTEX_IMB:
            vertex_imb = self.get_vertex_imbalance(partitioning)
            metrics["VERTEX_IMB"] = vertex_imb

        return metrics

    def report(self, visualize=True, barh=True): # TODO: include plotting parameters 
        """Collect individual reports for every partitioning.

        Parameters
        ----------
        visualize [True/False] flag indicating whether to visualize output

        Returns
        -------
        reports: calculated metrics for all partitionings, dictionary with key
        as numbers of partitions and values as dictionary with metrics
        """
        reports = {}
        for name, partitioning in self.partitionings.items():
            reports[name] = self.report_single_partitioning(partitioning, EDGE_IMB=True, \
                                                         VERTEX_IMB=True)

        if visualize:
            plt.figure(figsize=(15,10))
            ind = 1
            row_size = 3
            size = int(len(reports[list(reports.keys())[0]]) / row_size) + 1
            for metric in reports[list(reports.keys())[0]]:
                plot = False
                dat = []
                for report in reports:
                    if reports[report][metric] != None:
                        dat.append((str(report),reports[report][metric]))
                        plot = True
                if plot:
                    plt.subplot(size, row_size, ind)
                    if barh:
                        plt.barh(*list(zip(*dat)))
                    else:
                        plt.bar(*list(zip(*dat)))

                    plt.title(metric)
                    ind += 1
            plt.show()

        return reports

def main():
    pass

if __name__ == "__main__":
    main()
