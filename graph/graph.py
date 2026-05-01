import json
import networkx as nx
from graph.node import Node, NodeType
from graph.edge import Edge


class StemGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}

    def add_node(self, node: Node):
        self.nodes[node.id] = node
        self.graph.add_node(node.id)

    def add_edge(self, edge: Edge):
        source = self.nodes.get(edge.source_id)
        target = self.nodes.get(edge.target_id)

        if source is None or target is None:
            raise ValueError("Both source and target nodes must exist before adding an edge")
        
        self.edges[edge.id] = edge
        self.graph.add_edge(edge.source_id, edge.target_id, relation=edge.relation)

    def get_node(self, node_id: str) -> Node:
        return self.nodes.get(node_id)
    
    def get_traversal_order(self) -> list[str]:
        return list(nx.topological_sort(self.graph))
    
    def to_dict(self) -> dict:
        return {
            "nodes": [n.model_dump() for n in self.nodes.values()],
            "edges": [e.model_dump() for e in self.edges.values()]
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_inference_dict(self) -> dict:
        """
        The goal is to have a graph with only nodes with roles domain_knowledge, statgy and tool
        """
        inference_roles = {"domain_knowledge", "strategy", "tool"}
        inference_ids = {nid for nid, n in self.nodes.items() if n.role.value in inference_roles}
        return {
            "nodes": [n.model_dump() for nid, n in self.nodes.items() if nid in inference_ids],
            "edges": [e.model_dump() for e in self.edges.values()
                      if e.source_id in inference_ids and e.target_id in inference_ids]
        }

    def is_connected(self) -> bool:
        return nx.is_weakly_connected(self.graph)
    
    def get_isolated_nodes(self) ->  list[str]:
        return [n for n in self.graph.nodes if self.graph.degree(n) == 0]
    
    @classmethod
    def from_dict(cls, data: dict) -> "StemGraph":
        g = cls()

        for n in data["nodes"]:
            g.add_node(Node(**n))

        for e in data["edges"]:
            g.add_edge(Edge(**e))

        return g
