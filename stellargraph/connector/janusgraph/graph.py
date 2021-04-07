# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ["JanusGraphStellarGraph", "JanusGraphStellarDiGraph"]

import time
import diskcache as dc
import numpy as np
from ... import globalvar
from ...core.experimental import experimental

from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=8)

cache = dc.FanoutCache( "./cache_dir/features", shards=16, eviction_policy='least-frequently-used' )

@experimental(reason="the class is not tested", issues=[1578])
class JanusGraphStellarGraph:
    """
    JanusGraphStellarGraph class for graph machine learning on graphs stored in
    a JanusGraph database.

    This class communicates with JanusGramph via a python-gremlin traversal instance
    connected to the graph database of interest and contains functions to query the graph
    data necessary for machine learning.

    Args:
        graph_traversal (py2neo.Graph): a Python-Gremlin graph traversal object.
        node_label (str, optional): Common label for all nodes in the graph, if such label exists.
            Providing this is useful if there are any indexes created on this label (e.g. on node IDs),
            as it will improve performance of queries.
        id_property (str, optional): Name of JanusGraph property to use as ID.
        features_property (str, optional): Name of JanusGraph property to use as features.
        is_directed (bool, optional): If True, the data represents a
            directed multigraph, otherwise an undirected multigraph.
    """

    def __init__(
        self,
        graph_traversal,
        node_label=None,
        id_property="id",
        features_property="feature",
        is_directed=False,
    ):

        self.g = graph_traversal

        self.node_label = node_label
        self._is_directed = is_directed
        self._node_feature_size = None

        # names of properties to use when querying the database
        self.id_property = id_property
        self.features_property = features_property

        # FIXME: methods in this class currently only support homogeneous graphs with default node type
        self._node_type = globalvar.NODE_TYPE_DEFAULT

    def nodes(self):

        return self.g.V().values(self.id_property).toList()

    def node_feature_sizes(self):
        """
        Get the feature sizes for the node types in the graph.

        This method obtains the feature size by sampling a random node from the graph. Currently
        this class only supports a single default node type, and makes the following assumptions:

        - all nodes have features as a single list

        - all nodes' features have the same size

        - there's no mutations that change the size(s)

        Returns:
            A dictionary of node type and integer feature size.
        """
        if self._node_feature_size is None:

            self._node_feature_size = self.g.V().limit(1).values(self.features_property).count().next()

        return {self._node_type: self._node_feature_size}

    def is_directed(self):

        return self._is_directed

    def unique_node_type(self, error_message=None):
        """
        Return the unique node type, for a homogeneous-node graph.

        Args:
            error_message (str, optional): a custom message to use for the exception; this can use
                the ``%(found)s`` placeholder to insert the real sequence of node types.

        Returns:
            If this graph has only one node type, this returns that node type, otherwise it raises a
            ``ValueError`` exception.
        """
        return self._node_type

    def _pull_node_features(self, node_id):

        if ( node_id in cache ):

            return cache[node_id]

        feature = self.g.V().has(self.id_property, node_id).values(self.features_property).toList()

        cache.add( node_id, feature )

        return feature

    def node_features(self, nodes):
        """
        Get the numeric feature vectors for the specified nodes or node type.

        Args:
            nodes (list or hashable, optional): Node ID or list of node IDs.
        Returns:
            Numpy array containing the node features for the requested nodes.
        """

        # None's should be filled with zeros in the feature matrix

        #t1 = time.time()

        #features = np.zeros(
        #        (len(nodes), self.node_feature_sizes()[self._node_type])
        #        )

        #nodes = np.array( nodes )
        #valid = nodes[ nodes != None ]

        #unique = set( valid )

        #for node_id in unique:

        #    features[ nodes == node_id ] = self._pull_node_features(node_id)

        #t2 = time.time()

        #print( f"\n\nfeature: {t2-t1:1.4E}\n\n")

        #return features

        t1 = time.time()

        features = np.zeros( (len(nodes), self.node_feature_sizes()[self._node_type]) )

        nodes = np.array( nodes )
        valid = nodes[ nodes != None ]

        unique = set( valid )

        unique_futures = [ executor.submit( self._pull_node_features, node_id) for node_id in unique ]
        unique_features = [ f.result() for f in unique_futures ]

        for node_id, feature in zip( unique, unique_features ):

            features[ nodes == node_id ] = feature

        t2 = time.time()

        #print( f"features: {t2-t1:1.4E}")

        return features

# A convenience class that merely specifies that edges have direction.
class JanusGraphStellarDiGraph(JanusGraphStellarGraph):
    def __init__(
        self,
        graph_traversal,
        node_label=None,
        id_property="id",
        features_property="feature",
    ):
        super().__init__(
            graph_traversal,
            node_label=node_label,
            id_property=id_property,
            features_property=features_property,
            is_directed=True,
        )
