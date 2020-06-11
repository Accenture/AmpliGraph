# Copyright 2020 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import copy

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
            edge_cut_proportion = (edge_cut * 100) / avg_size  # edge cut with respect to the average partition size
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

        edge_imb = max_size / avg_size - 1
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
            lengths.append(len(np.asarray(list(set(partition[:, 0]).union(set(partition[:, 2]))))))

        vertex_imb = np.max(lengths) / np.mean(lengths) - 1
        return vertex_imb

    def get_average_deviation_from_ideal_size_vertices(self, partitions):
        """Metric that calculates the average difference between the
           ideal size of partition (in terms of vertices) and real,
           it is expressed as percentage deviation from ideal size.

           Parameters
           ----------
           partitions: partitions in one partitioning

           Returns
           -------
           percentage_dev: percentage vertex size partition deviation
        """
        k = len(partitions)
        vertices = set()
        sizes = []
        for partition in partitions:
            tmp = set(partition[:,0])
            tmp.update(set(partition[:,2]))
            vertices.update(tmp)
            sizes.append(len(tmp))
        data_size = len(vertices)
        ideal_size = data_size/k
        percentage_dev = ((np.sum([np.abs(ideal_size - size) for size in sizes])/k)/ideal_size)*100
        return percentage_dev

    def get_average_deviation_from_ideal_size_edges(self, partitions):
        """Metric that calculates the average difference between the
           ideal size of partition (in terms of edges) and real,
           it is expressed as percentage deviation from ideal size.

           Parameters
           ----------
           partitions: partitions in one partitioning

           Returns
           -------
           percentage_dev: percentage edge size partition deviation
        """

        k = len(partitions)
        edges = set()
        sizes = []
        for partition in partitions:
            edges.update(set(["{} {} {}".format(*list(e)) for e in partition]))
            sizes.append(len(partition))
        data_size = len(edges)
        ideal_size = data_size/k
        percentage_dev = ((np.sum([np.abs(ideal_size - size) for size in sizes])/k)/ideal_size)*100
        return percentage_dev

    def get_vertex_count(self, partitions):
        """Counts number of vertices in each partition 
           that estimates the size of partition.
        
           Parameters
           ---------
           partitions: partitions in one partitioning

           Returns
           -------
           info: list of counts, e.g. for 2 partitions the list will be of size two with
                 vertex count for each partition - (5,6), for 3 partitions:(5,2,4). 
        """
        info = []
        for partition in partitions:
            vertices = len(set(partition[:, 0]).union(set(partition[:, 2])))
            info.append(vertices)

        return info 

    def get_edges_count(self, partitions):
        """Counts number of edges in each partition that 
           estimates the size of partition.
        
           Parameters
           ---------
           partitions: partitions in one partitioning

           Returns
           -------
           info: list of counts, e.g. for 2 partitions the list will be of size two with
                 the edge count - (10,12), for 3 partitions: (7,8,7). 
        """
        info = []
        for partition in partitions:
            edges = len(partition)
            info.append(edges)

        return info        

    def get_modularity(self):
        """Calculates modularity of partitions.

        Parameters
        ----------

        Returns
        -------
        modularity: modularity
        """
        raise NotImplementedError

    def report_single_partitioning(self, partitioning, EDGE_CUT=True, 
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
        metrics["VERTEX_COUNT"] = self.get_vertex_count(partitioning)
        metrics["EDGES_COUNT"] = self.get_edges_count(partitioning)
        metrics["PERCENTAGE_DEV_EDGES"] = self.get_average_deviation_from_ideal_size_edges(partitioning)
        metrics["PERCENTAGE_DEV_VERTICES"] = self.get_average_deviation_from_ideal_size_vertices(partitioning)

        return metrics

    def report(self, visualize=True, barh=True):  # TODO: include plotting parameters 
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
            reports[name] = self.report_single_partitioning(partitioning, EDGE_IMB=True,
                                                            VERTEX_IMB=True)

        if visualize:
            plt.figure(figsize=(15,10))
            ind = 1
            row_size = 3
            size = int(len(reports[list(reports.keys())[0]]) / row_size) + 1
            for metric in reports[list(reports.keys())[0]]:
                plot = False
                dat = []
                color=iter(cm.PiYG(np.linspace(0,1,len(reports))))
                colors_aggregate = {r:next(color) for r in reports}                
                for j, report in enumerate(reports):
                    if reports[report][metric] is not None:
                        if type(reports[report][metric]) is list:
                            n = len(reports[report][metric])
                            color=iter(cm.seismic(np.linspace(0,1,n)))
                            colors = {'partition {}'.format(i):next(color) for i in range(n)}
                            width = 0.8/n
                            for i, r in enumerate(reports[report][metric]):
                                label = 'partition {}'.format(i)
                                dat.append({'y':j + (i*width), 'width': r, "height": width, 
                                            'label':label, 'label2':str(report), "color":colors[label]})
                        else:
                            colors = colors_aggregate
                            label = str(report)
                            dat.append({"y":j, "width":reports[report][metric], 
                                        'label2':label, 'color':colors[label]})
                        plot = True
                if plot:
                    plt.subplot(size, row_size, ind) 
                    
                    if barh:
                        unpacked = {k: [dic[k] for dic in dat] for k in dat[0]}
                        data = copy.deepcopy(unpacked)
                        del unpacked["label2"]
                        plt.barh(**unpacked, edgecolor='white')
                    else:
                        plt.bar(*list(zip(*dat)), edgecolor='white')
                        
                    labels = list(colors.keys())
                    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
                    #plt.legend(handles, labels)                        
                    labels = set(list(data["label2"]))
                    if type(labels) == set:
                        plt.yticks(range(len(list(labels))), list(labels))
                    plt.title(metric)
                    ind += 1                    
            plt.show()
            
        return reports


def compare_partitionings(list_of_partitioners, data, visualize=True):
    """Wrapper around PartitioningReporter hiding logging settings.

       Parameters
       ---------
       list_of_partitioners: list of uninitialized partitioners
       data: numpy array with graoh to be splited into partitions
       visualize [default=True]: flag whether to visualize results or not
    
       Returns
       -------
       result: dictionary with metrics evaluating partitionings
    
       Example
       -------
       >>>partitioners = [NaiveGraphPartitioner, 
                          SortedEdgesGraphPartitioner, 
                          DoubleSortedEdgesGraphPartitioner]
       >>>report = compare_partitionings(partitioners)
    """
    partitionings = {}    
    for partitioner in list_of_partitioners:
        logs = {}
        partitioner_fitted = partitioner(data, k=2)
        tmp = partitioner_fitted.split(log=logs)
        partitionings[partitioner.__name__] = (tmp, logs)
    reporter = PartitioningReporter(partitionings=partitionings)
    result = reporter.report(visualize=visualize, barh=True)
    return result


def main():
    dummy_partitionings = {"one": (np.array([[(0,1,2),
                                              (0,1,3),
                                              (0,1,4)],
                                             [(2,1,6),
                                              (6,1,4),
                                              (3,1,5)]]).astype(int),
                                              {"SPLIT": {"time": 10,"memory-bytes": 12}}),
                                              "two": (np.array([[(0,1,2),
                                              (0,1,3),
                                              (0,1,4)],[(2,1,6),
                                              (6,1,4),
                                              (3,1,5)]]).astype(int),
                                              {"SPLIT": {"time": 20,"memory-bytes": 10}})}

    reporter = PartitioningReporter(partitionings=dummy_partitionings)
    result = reporter.report(visualize=False)
    print(result)

if __name__ == "__main__":
    main()
