from typing import List, Set, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json

@dataclass
class Node:
    id: int
    label: str
    node_type: str  # 'token', 'property', 'merged_sequence', 'merged_aggregation', or 'sink', these helps GNN distinguish between code tokens and structural elements
    original_labels: List[str] = field(default_factory=list)  # For merged nodes, these are used in compression process to average out all original node embeddings
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id
    
    def __repr__(self):
        return f"Node({self.id}, {self.label}, {self.node_type})"


@dataclass
class Edge: # This is a directed edge
    source: Node
    target: Node
    edge_type: str = 'default'  # 'ast', 'next_token', 'to_sink', 'ast' represents reversed AST edges during construction of AG, 'next_token' represents dependency ordering which follows source code ordering, 'to_sink' connects all property nodes to the sink node which is root node.
    
    def __hash__(self):
        return hash((self.source.id, self.target.id))
    
    def __eq__(self, other):
        return isinstance(other, Edge) and self.source.id == other.source.id and self.target.id == other.target.id
    
    def __repr__(self):
        return f"Edge({self.source.label} -> {self.target.label})"


@dataclass
class AbstractGraph:
    """Abstract Graph (AG) representation"""
    nodes: Set[Node] = field(default_factory=set)
    edges: Set[Edge] = field(default_factory=set)
    sink: Optional[Node] = None
    token_nodes: List[Node] = field(default_factory=list)
    property_nodes: Set[Node] = field(default_factory=set)
    
    def get_incoming_edges(self, node: Node) -> List[Edge]:
        """Get all edges pointing to this node"""
        return [e for e in self.edges if e.target == node]
    
    def get_outgoing_edges(self, node: Node) -> List[Edge]:
        """Get all edges from this node"""
        return [e for e in self.edges if e.source == node]
    
    def get_in_degree(self, node: Node) -> int:
        """Count incoming edges"""
        return len(self.get_incoming_edges(node))
    
    def get_out_degree(self, node: Node) -> int:
        """Count outgoing edges"""
        return len(self.get_outgoing_edges(node))


@dataclass
class CompactAbstractGraph:
    """Compact Abstract Graph (CAG) representation"""
    nodes: Set[Node] = field(default_factory=set)
    edges: Set[Edge] = field(default_factory=set)
    sink: Optional[Node] = None
    
    def to_adjacency_list(self) -> Dict[int, List[int]]:
        """Convert to adjacency list format for GNN"""
        adj_list = defaultdict(list)
        for edge in self.edges:
            adj_list[edge.source.id].append(edge.target.id)
        return dict(adj_list)
    
    def get_node_features(self) -> Dict[int, Dict[str, Any]]:
        """Extract node features for GNN embedding"""
        features = {}
        for node in self.nodes:
            features[node.id] = {
                'label': node.label,
                'type': node.node_type,
                'original_labels': node.original_labels if node.original_labels else [node.label]
            }
        return features
    def save_to_file(self, filepath: str):
        """Save CAG to a JSON file for generating graph embeddings"""
        data = {
            'nodes': [{'id': node.id, 'label': node.label, 'type': node.node_type, 'original_labels': node.original_labels} for node in self.nodes],
            'edges': [{'source': edge.source.id, 'target': edge.target.id, 'type': edge.edge_type} for edge in self.edges],
            'sink': self.sink.id if self.sink else None
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)