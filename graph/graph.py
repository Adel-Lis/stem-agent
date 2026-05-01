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
        """Adds a node to both the internal dictionary and the underlying networkx graph."""
        self.nodes[node.id] = node
        self.graph.add_node(node.id)

    def add_edge(self, edge: Edge):
        """Connects two existing nodes with an edge, raising an error if either node doesn't exist yet."""
        source = self.nodes.get(edge.source_id)
        target = self.nodes.get(edge.target_id)

        if source is None or target is None:
            raise ValueError("Both source and target nodes must exist before adding an edge")

        self.edges[edge.id] = edge
        self.graph.add_edge(edge.source_id, edge.target_id, relation=edge.relation)

    def get_node(self, node_id: str) -> Node:
        """Returns the node with the given id, or None if it doesn't exist."""
        return self.nodes.get(node_id)

    def get_traversal_order(self) -> list[str]:
        """Returns node ids sorted so that every node comes after all the nodes it depends on."""
        return list(nx.topological_sort(self.graph))

    def to_dict(self) -> dict:
        """Converts the full graph to a plain dictionary, including all nodes and edges."""
        return {
            "nodes": [n.model_dump() for n in self.nodes.values()],
            "edges": [e.model_dump() for e in self.edges.values()]
        }

    def to_json(self) -> str:
        """Converts the full graph to a JSON string, used to show the builder the current graph state."""
        return json.dumps(self.to_dict(), indent=2)

    def to_inference_dict(self) -> dict:
        """Exports only the nodes the agent actually uses at run time — strips out the builder and evaluator genesis nodes before saving a checkpoint."""
        inference_roles = {"domain_knowledge", "strategy", "tool"}
        inference_ids = {nid for nid, n in self.nodes.items() if n.role.value in inference_roles}
        return {
            "nodes": [n.model_dump() for nid, n in self.nodes.items() if nid in inference_ids],
            "edges": [e.model_dump() for e in self.edges.values()
                      if e.source_id in inference_ids and e.target_id in inference_ids]
        }

    def is_connected(self) -> bool:
        """Returns True if every node in the graph is reachable from every other node."""
        return nx.is_weakly_connected(self.graph)

    def get_isolated_nodes(self) ->  list[str]:
        """Returns a list of node ids that have no edges at all — used to catch nodes the builder accidentally left disconnected."""
        return [n for n in self.graph.nodes if self.graph.degree(n) == 0]

    @classmethod
    def from_dict(cls, data: dict) -> "StemGraph":
        """Rebuilds a StemGraph from a plain dictionary, used when loading a saved checkpoint."""
        g = cls()

        for n in data["nodes"]:
            g.add_node(Node(**n))

        for e in data["edges"]:
            g.add_edge(Edge(**e))

        return g
