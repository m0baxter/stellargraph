# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = [
    "JanusGraphSampledBreadthFirstWalk",
    "JanusGraphDirectedBreadthFirstNeighbors",
]

import time
import diskcache as dc
import numpy as np
from ...core.experimental import experimental
from .graph import JanusGraphStellarGraph, JanusGraphStellarDiGraph

from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=8)

cache = dc.FanoutCache( "./cache_dir/neighbours", shards=16, eviction_policy='least-frequently-used' )

def _bfs_neighbor_query( g, sampling_direction, id_property, num_samples, node_id, node_label=None):

    if ( node_id == None ):

        return num_samples * [ None ]

    if ( node_id in cache ):

        res = cache[ node_id ]

    else:

        traversal = g.V()

        if node_label:

            traversal = traversal.hasLabel(node_label)

        traversal = traversal.has( id_property, node_id )

        if ( sampling_direction == "BOTH" ):

            traversal = traversal.bothE()

        elif ( sampling_direction == "IN" ):

            traversal = traversal.inE()

        elif ( sampling_direction == "OUT" ):

            traversal = traversal.outE()

        res = traversal = traversal.limit(1000).otherV().values(id_property).toList()

        if ( len(res) == 0 ):

            return num_samples * [ None ]

    cache.add( node_id, res )

    res = np.random.choice( res, size = num_samples, replace = True ).tolist()

    return res

@experimental(reason="the class is not fully tested")
class JanusGraphSampledBreadthFirstWalk:
    """
    Breadth First Walk that generates a sampled number of paths from a starting node.
    It can be used to extract a random sub-graph starting from a set of initial nodes from Neo4j database.
    """

    def __init__(self, graph):
        if not isinstance(graph, JanusGraphStellarGraph):
            raise TypeError("Graph must be a JanusGraphStellarGraph or JanusGraphStellarDiGraph.")
        self.graph = graph

    def run(self, nodes=None, n=1, n_size=None):
        """
        Send queries to Neo4j graph databases and collect sampled breadth-first walks starting from
        the root nodes.

        Args:
            nodes (list of hashable): A list of root node ids such that from each node n BFWs will
                be generated up to the given depth d.
            n_size (list of int): The number of neighbouring nodes to expand at each depth of the
                walk. Sampling of neighbours with replacement is always used regardless of the node
                degree and number of neighbours requested.
            n (int): Number of walks per node id.
            seed (int, optional): Random number generator seed; default is None

        Returns:
            A list of lists, each list is a sequence of sampled node ids at a certain hop.
        """
        t1 = time.time()

        samples = [[head_node for head_node in nodes for _ in range(n)]]

        #for num_samples in n_size:

        #    cur_nodes = samples[-1]
        #    new_results = []

        #    for node_id in cur_nodes:

        #        tmp = _bfs_neighbor_query(
        #                self.graph.g,
        #                sampling_direction = "BOTH",
        #                id_property = self.graph.id_property,
        #                num_samples = num_samples,
        #                node_id = node_id,
        #                node_label = self.graph.node_label,
        #                )

        #        new_results.extend( tmp )

        #    samples.append( new_results )

        for num_samples in n_size:

            cur_nodes = samples[-1]
            new_results = []

            sample_futures = [ executor.submit(
                _bfs_neighbor_query,
                self.graph.g,
                "BOTH",
                self.graph.id_property,
                num_samples,
                node_id,
                self.graph.node_label,
                ) for node_id in cur_nodes ]

            new_results = [ f.result() for f in sample_futures ]

            samples.append( sum( new_results, [] ) )

        t2 = time.time()

        #print( f"sample: {t2-t1:1.4E}" )

        return samples

@experimental(reason="the class is not fully tested")
class JanusGraphDirectedBreadthFirstNeighbors:
    """
    Breadth First Walk that generates a sampled number of paths from a starting node.
    It can be used to extract a random sub-graph starting from a set of initial nodes from JanusGraph database.
    """

    def __init__(self, graph):
        if not isinstance(graph, JanusGraphStellarDiGraph):
            raise TypeError("Graph must be a JanusGraphStellarDiGraph.")
        self.graph = graph

    def run(self, nodes=None, n=1, in_size=None, out_size=None):
        """
        Send queries to Neo4j databases and collect sampled breadth-first walks starting from the root nodes.

        Args:
            nodes (list of hashable): A list of root node ids such that from each node n BFWs will
                be generated up to the given depth d.
            n (int): Number of walks per node id.
            in_size (list of int): The number of in-directed nodes to sample with replacement at each depth of the walk.
            out_size (list of int): The number of out-directed nodes to sample with replacement at each depth of the walk.
        Returns:
            A list of multi-hop neighbourhood samples. Each sample expresses a collection of nodes, which could be either in-neighbors,
            or out-neighbors of the previous hops.
            Result has the format:
            [[head1, head2, ...],
            [in1_head1, in2_head1, ..., in1_head2, in2_head2, ...], [out1_head1, out2_head1, ..., out1_head2, out2_head2, ...],
            [in1_in1_head1, in2_in1_head1, ..., in1_in2_head1, ...], [out1_in1_head1, out2_in1_head1, ..., out1_in2_head1, ...],
            [in1_out1_head1, in2_out1_head1, ..., in1_out2_head1, ...], [out1_out1_head1, out2_out1_head1, ..., out1_out2_head1, ...],
            ...
            ]
        """
        # FIXME: we may want to run validation on all the run parameters similar to other GraphWalk classes

        head_nodes = [head_node for head_node in nodes for _ in range(n)]
        hops = [[head_nodes]]

        for in_num, out_num in zip( in_size, out_size ):

            last_hop = hops[-1]
            this_hop = []

            for cur_nodes in last_hop:

                in_samples = []
                out_samples = []

                for node_id in cur_nodes:

                    in_tmp = _bfs_neighbor_query(
                            self.graph.g,
                            sampling_direction="IN",
                            id_property=self.graph.id_property,
                            num_samples=in_size,
                            node_id = node_id,
                            node_label=self.graph.cypher_node_label,
                            )

                    out_tmp = _bfs_neighbor_query(
                            self.graph.g,
                            sampling_direction="OUT",
                            id_property=self.graph.id_property,
                            num_samples=out_size,
                            node_id = node_id,
                            node_label=self.graph.cypher_node_label,
                            )

                    in_samples.extend( in_tmp )
                    out_samples.extend( out_tmp )

                this_hop.append( in_samples )
                this_hop.append( out_samples )

            hop.append( this_hop )

        return sum(hops, [])
