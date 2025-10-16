from typing import List, Dict, Tuple, Any
from datautils.classes import AbstractGraph, CompactAbstractGraph
from utils.graphutils import CCodeCAGBuilder

class CAGStatistics:
    """Calculate and compare statistics for AG vs CAG"""
    
    @staticmethod
    def calculate_compression_ratio(ag: AbstractGraph, cag: CompactAbstractGraph) -> Dict[str, Any]:
        """Calculate compression statistics"""
        ag_nodes = len(ag.nodes)
        ag_edges = len(ag.edges)
        cag_nodes = len(cag.nodes)
        cag_edges = len(cag.edges)
        
        return {
            'ag_nodes': ag_nodes,
            'ag_edges': ag_edges,
            'cag_nodes': cag_nodes,
            'cag_edges': cag_edges,
            'node_reduction_count': ag_nodes - cag_nodes,
            'edge_reduction_count': ag_edges - cag_edges,
            'node_reduction_percent': ((ag_nodes - cag_nodes) / ag_nodes * 100) if ag_nodes > 0 else 0,
            'edge_reduction_percent': ((ag_edges - cag_edges) / ag_edges * 100) if ag_edges > 0 else 0
        }
    
    @staticmethod
    def print_statistics(stats: Dict[str, Any], func_name: str = ""):
        """Print formatted statistics"""
        print(f"\n{'='*70}")
        print(f"Compression Statistics" + (f" for {func_name}" if func_name else ""))
        print('='*70)
        print(f"Abstract Graph (AG):")
        print(f"  Nodes: {stats['ag_nodes']:,}")
        print(f"  Edges: {stats['ag_edges']:,}")
        print(f"\nCompact Abstract Graph (CAG):")
        print(f"  Nodes: {stats['cag_nodes']:,}")
        print(f"  Edges: {stats['cag_edges']:,}")
        print(f"\nReduction:")
        print(f"  Nodes removed: {stats['node_reduction_count']:,} ({stats['node_reduction_percent']:.2f}%)")
        print(f"  Edges removed: {stats['edge_reduction_count']:,} ({stats['edge_reduction_percent']:.2f}%)")


class CAGDatasetBuilder:
    """Build datasets of CAGs for training GNN models"""
    
    def __init__(self, builder: CCodeCAGBuilder):
        self.builder = builder
        self.cags: List[Tuple[str, CompactAbstractGraph, int]] = []  # (name, cag, label)
    
    def add_code_sample(self, code: str, label: int, sample_name: str = None):
        """
        Add a code sample to the dataset
        
        Args:
            code: C source code
            label: Vulnerability label (0 = safe, 1 = vulnerable)
            sample_name: Optional name for the sample
        """
        try:
            cags = self.builder.build_cag_from_code(code)
            for func_name, cag in cags.items():
                name = sample_name or func_name
                self.cags.append((name, cag, label))
        except Exception as e:
            print(f"Warning: Failed to process sample {sample_name}: {e}")
    
    def add_from_file(self, filepath: str, label: int):
        """Add all functions from a C file"""
        with open(filepath, 'r') as f:
            code = f.read()
        self.add_code_sample(code, label, filepath)
    
    def export_for_gnn(self, output_file: str = None) -> Dict:
        """
        Export dataset in format ready for GNN training
        
        Returns format compatible with PyTorch Geometric
        """
        dataset = {
            'graphs': [],
            'labels': [],
            'metadata': []
        }
        
        for name, cag, label in self.cags:
            graph_data = {
                'name': name,
                'num_nodes': len(cag.nodes),
                'num_edges': len(cag.edges),
                'node_features': cag.get_node_features(),
                'edge_index': [(e.source.id, e.target.id) for e in cag.edges],
                'adjacency_list': cag.to_adjacency_list()
            }
            
            dataset['graphs'].append(graph_data)
            dataset['labels'].append(label)
            dataset['metadata'].append({
                'name': name,
                'num_nodes': len(cag.nodes),
                'num_edges': len(cag.edges)
            })
        
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(dataset, f, indent=2)
        
        return dataset
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        total_nodes = sum(len(cag.nodes) for _, cag, _ in self.cags)
        total_edges = sum(len(cag.edges) for _, cag, _ in self.cags)
        labels = [label for _, _, label in self.cags]
        
        return {
            'num_samples': len(self.cags),
            'num_vulnerable': sum(labels),
            'num_safe': len(labels) - sum(labels),
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'avg_nodes_per_graph': total_nodes / len(self.cags) if self.cags else 0,
            'avg_edges_per_graph': total_edges / len(self.cags) if self.cags else 0
        }
