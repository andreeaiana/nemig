import networkx as nx
from itertools import chain
from collections import defaultdict
from typing import Set, List, Dict, Tuple, Union


class BaseGraph():
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    @property
    def nodes(self) -> Set:
        return set(self.graph.nodes)

    def get_nodes(self, data=False) -> List:
        return list(self.graph.nodes(data))

    def has_node(self, node: str) -> bool:
        return node in self.graph

    def _add_nodes(self, nodes: List[str]) -> None:
        self.graph.add_nodes_from(nodes)

    def _remove_nodes(self, nodes: List[str]) -> None:
        self.graph.remove_nodes_from(nodes)

    @property
    def edges(self) -> Set:
        return set(self.graph.edges)
    
    def get_edges(self, nbunch=None, data=False, keys=False) -> List: 
        return list(self.graph.edges(nbunch, data, keys))

    def has_edge(self, edge: Union[Tuple[str, str], Tuple[str, str, str]]) -> bool:
        """ Checks if the graph has an edge (u, v) or (u, v, k). """
        if len(edge) == 2:
            return self.graph.has_edge(edge[0], edge[1])
        else:
            return self.graph.has_edge(edge[0], edge[1], key=edge[2])

    def _add_edge(self, edge: Tuple[str, str], key: str) -> None:
        self.graph.add_edge(edge[0], edge[1], key)

    def _add_edges(self, edges: List[Tuple[str, str, str]]) -> None:
        """ Add edges of the form (u, v, k). """
        self.graph.add_edges_from(edges)
    
    def _remove_edge(self, edge: Union[Tuple[str, str], Tuple[str, str, str]], key: str=None) -> None:
        """ Removes edge of type (u, v) or (u, v, k). """
        self.graph.remove_edge(edge, key)

    def _remove_edges(self, edges: List[Union[Tuple[str, str], Tuple[str, str, str]]]) -> None:
        """ Removes edges of type (u, v) or (u, v, k). """
        self.graph.remove_edges_from(edges)

    def parents(self, node: str) -> Set:
        return set(self.graph.predecessors(node)) if self.graph.has_node(node) else set()

    def ancestors(self, node: str) -> Set:
        return set(nx.ancestors(self.graph, node)) if self.graph.has_node(node) else set()

    def children(self, node: str) -> Set:
        return set(self.graph.successors(node)) if self.graph.has_node(node) else set()

    def descendants(self, node: str) -> Set:
        return set(nx.descendants(self.graph, node)) if self.graph.has_node(node) else set()

    def depth(self, source_node: str, target_node: str) -> int:
        """ Returns the length of the shortest path between two given nodes (or -1 if none exists). """
        try:
            return nx.shortest_path_length(self.graph, source=source_node, target=target_node)
        except nx.NetworkXNoPath:
            return -1

    def depths(self, source_node: str) -> Dict:
        """ Returns the lengths of the shortest path from all nodes of the graph to the given node. """
        return defaultdict(lambda: -1, nx.shortest_path_length(self.graph, source=source_node))
    
    def is_unconnected(self, node: str) -> bool:
        """ Checks if node is connected to the rest of the graph. """
        if not self.graph.has_node(node):
            raise Exception(f'Node {node} not in graph')
        
        source_nodes = [edge[0] for edge in self.graph.edges]
        target_nodes = [edge[1] for edge in self.graph.edges]
        
        return node in source_nodes or node in target_nodes

    def is_multigraph(self) -> bool:
        return True

    def remove_unconnected(self) -> None:
        """ Removes all nodes that are not connected to any other node. """
        unconnected_nodes = [node for node in self.graph.nodes if self.is_unconnected(node)]
        self._remove_nodes(unconnected_nodes)

    def _get_attr(self, node: str, attr: str):
        return self.graph.nodes(data=attr)[node]

    def _set_attr(self, node: str, attr: str, val: str) -> None:
        self.graph.nodes[node][attr] = val

    def _update_attr(self, node: str, attr: str, val: str) -> None:
        """ Updates an attribute set."""
        if type(self.graph.nodes[node][attr]) == set:
            if type(val) == set:
                self.graph.nodes[node][attr].update(val)
            else:
                self.graph.nodes[node][attr].add(val)
        else:
            self.graph.nodes[node][attr] = val

    def _remove_attr_val(self, node: str, attr: str, val: str) -> None:
        """ Removes a value from an attribute list. """
        self.graph.nodes[node][attr].remove(val)

    def _reset_attr(self, node: str, attr: str) -> None:
        if attr in self.graph.nodes[node]:
            del self.graph.nodes[node][attr]

    def _set_edge_attr(self, edge: Tuple[str, str], attr: str, val: str) -> None:
        self.graph[edge[0]][edge[1]][attr] = val

    def _reset_edge_attr(self, edge: Tuple[str, str], attr: str) -> None:
        if attr in self.graph.edges[edge]:
            del self.graph.edges[edge][attr]

    def number_of_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def number_of_edges(self) -> int:
        return self.graph.number_of_edges()
    
    def degree(self, nbunch=None):
        return self.graph.degree(nbunch)

    def in_degree(self, nbunch=None):
        return self.graph.in_degree(nbunch)

    def out_degree(self, nbunch=None):
        return self.graph.out_degree(nbunch)

    def neighbors(self, node: str) -> set:
        return set(self.graph.neighbors(node))

    def is_directed(self) -> bool:
        return self.graph.is_directed()

    def in_edges(self, nbunch=None, data=False, keys=False, default=None):
        return self.graph.in_edges(nbunch, data, keys, default)

    def out_edges(self, nbunch=None, data=False, keys=False, default=None):
        return self.graph.out_edges(nbunch, data, keys, default)

    def contracted_nodes(self, u, v, attr2update: List, self_loops=True) -> None:
        # Based on https://networkx.org/documentation/stable/_modules/networkx/algorithms/minors/contraction.html#contracted_nodes

        edges_to_remap = chain(self.in_edges(v, keys=True), self.out_edges(v, keys=True))
        edges_to_remap = list(edges_to_remap)

        v_data = self.graph.nodes[v]
        self._remove_nodes([v])

        for (prev_w, prev_x, d) in edges_to_remap:
            w = prev_w if prev_w != v else u
            x = prev_x if prev_x != v else u

            if ({prev_w, prev_x} == {u,v}) and not self_loops:
                continue

            if not self.has_edge((w, x)) or self.is_multigraph():
                self._add_edge((w, x), d)
            
        # Merge the attributes of the merged nodes
        for attr in attr2update:
            if attr in v_data:
                if attr in self.graph.nodes[u]:
                    self._update_attr(u, attr, v_data[attr])
                else:
                    self._set_attr(u, attr, v_data[attr])

