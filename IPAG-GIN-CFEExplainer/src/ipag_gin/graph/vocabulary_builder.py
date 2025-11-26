# File: vocabulary_builder.py

import numpy as np
from collections import Counter
import pickle


class VocabularyBuilder:
    """
    Builds vocabularies from IPAG nodes without requiring storage in memory.
    Works with the streaming output of IPAGBuilder.
    """
    
    def __init__(self):
        self.node_type_vocab = None
        self.original_type_vocab = None
        self.label_vocab = None
        
        # Counters for building vocabularies
        self._node_type_counter = Counter()
        self._original_type_counter = Counter()
        self._label_counter = Counter()
        
        self._is_finalized = False
    
    def update(self, ipag_nodes):
        """
        Update vocabulary counters with nodes from one or more graphs.
        Call this as you process each graph.
        
        Args:
            ipag_nodes: List of node dicts from a single graph,
                       OR List of List of node dicts from multiple graphs
        """
        if self._is_finalized:
            raise RuntimeError("Vocabulary already finalized. Create new builder to update.")
        
        # Handle single graph or batch of graphs
        if ipag_nodes and isinstance(ipag_nodes[0], dict):
            # Single graph: list of node dicts
            self._update_single(ipag_nodes)
        else:
            # Batch: list of list of node dicts
            for nodes in ipag_nodes:
                if nodes:  # Skip empty graphs
                    self._update_single(nodes)
    
    def _update_single(self, nodes):
        """Update counters from a single graph's nodes."""
        for node in nodes:
            self._node_type_counter[node['type']] += 1
            self._original_type_counter[node['original_type']] += 1
            
            label = node.get('label', '')
            if label and label.strip():
                self._label_counter[label] += 1
    
    def finalize(self, min_freq=1):
        """
        Finalize vocabularies after processing all graphs.
        
        Args:
            min_freq (int): Minimum frequency to include in vocabulary.
                           Labels below this threshold become <UNK>.
        
        Returns:
            dict: Dictionary containing all vocabularies
        """
        print("\n" + "=" * 60)
        print("Finalizing Vocabularies")
        print("=" * 60)
        
        # Node type vocabulary (include all, they're structural)
        self.node_type_vocab = {'<PAD>': 0, '<UNK>': 1}
        for idx, (t, _) in enumerate(sorted(self._node_type_counter.items()), start=2):
            self.node_type_vocab[t] = idx
        
        # Original type vocabulary (include all)
        self.original_type_vocab = {'<PAD>': 0, '<UNK>': 1}
        for idx, (t, _) in enumerate(sorted(self._original_type_counter.items()), start=2):
            self.original_type_vocab[t] = idx
        
        # Label vocabulary (filter by frequency)
        self.label_vocab = {'<PAD>': 0, '<UNK>': 1, '<EMPTY>': 2}
        filtered_labels = [(l, c) for l, c in self._label_counter.items() if c >= min_freq]
        for idx, (l, _) in enumerate(sorted(filtered_labels), start=3):
            self.label_vocab[l] = idx
        
        # Create reverse mappings
        self.idx_to_node_type = {v: k for k, v in self.node_type_vocab.items()}
        self.idx_to_original_type = {v: k for k, v in self.original_type_vocab.items()}
        self.idx_to_label = {v: k for k, v in self.label_vocab.items()}
        
        self._is_finalized = True
        
        print(f"Node type vocabulary: {len(self.node_type_vocab)} types")
        print(f"Original type vocabulary: {len(self.original_type_vocab)} types")
        print(f"Label vocabulary: {len(self.label_vocab)} labels (min_freq={min_freq})")
        print(f"  Labels filtered out: {len(self._label_counter) - len(self.label_vocab) + 3}")
        print("=" * 60)
        
        return self.get_vocabularies()
    
    def get_vocabularies(self):
        """Return all vocabularies as a dictionary."""
        if not self._is_finalized:
            raise RuntimeError("Vocabularies not finalized. Call finalize() first.")
        
        return {
            'node_type_vocab': self.node_type_vocab,
            'original_type_vocab': self.original_type_vocab,
            'label_vocab': self.label_vocab,
            'idx_to_node_type': self.idx_to_node_type,
            'idx_to_original_type': self.idx_to_original_type,
            'idx_to_label': self.idx_to_label,
        }
    
    def get_vocab_sizes(self):
        """Return vocabulary sizes for model initialization."""
        if not self._is_finalized:
            raise RuntimeError("Vocabularies not finalized. Call finalize() first.")
        
        return {
            'node_type': len(self.node_type_vocab),
            'original_type': len(self.original_type_vocab),
            'label': len(self.label_vocab),
        }
    
    def save(self, path='vocabularies.pkl'):
        """Save vocabularies to file."""
        if not self._is_finalized:
            raise RuntimeError("Vocabularies not finalized. Call finalize() first.")
        
        with open(path, 'wb') as f:
            pickle.dump(self.get_vocabularies(), f)
        print(f"Vocabularies saved to {path}")
    
    @classmethod
    def load(cls, path='vocabularies.pkl'):
        """Load vocabularies from file."""
        builder = cls()
        
        with open(path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        builder.node_type_vocab = vocab_data['node_type_vocab']
        builder.original_type_vocab = vocab_data['original_type_vocab']
        builder.label_vocab = vocab_data['label_vocab']
        builder.idx_to_node_type = vocab_data['idx_to_node_type']
        builder.idx_to_original_type = vocab_data['idx_to_original_type']
        builder.idx_to_label = vocab_data['idx_to_label']
        builder._is_finalized = True
        
        print(f"Vocabularies loaded from {path}")
        print(f"  Node types: {len(builder.node_type_vocab)}")
        print(f"  Original types: {len(builder.original_type_vocab)}")
        print(f"  Labels: {len(builder.label_vocab)}")
        
        return builder


class NodeFeatureEncoder:
    """
    Encodes IPAG nodes into numerical features using pre-built vocabularies.
    Designed for streaming - encode graphs one at a time.
    """
    
    def __init__(self, vocabularies):
        """
        Initialize encoder with vocabularies.
        
        Args:
            vocabularies: dict from VocabularyBuilder.get_vocabularies()
                         OR VocabularyBuilder instance
        """
        if isinstance(vocabularies, VocabularyBuilder):
            vocabularies = vocabularies.get_vocabularies()
        
        self.node_type_vocab = vocabularies['node_type_vocab']
        self.original_type_vocab = vocabularies['original_type_vocab']
        self.label_vocab = vocabularies['label_vocab']
    
    def encode(self, ipag_nodes, use_one_hot=False):
        """
        Encode a single graph's nodes into features.
        
        Args:
            ipag_nodes: List of node dicts from one graph
            use_one_hot (bool): If True, use one-hot. If False, use indices.
        
        Returns:
            np.ndarray: Feature matrix [num_nodes, feature_dim]
        """
        if not ipag_nodes:
            if use_one_hot:
                total_size = (len(self.node_type_vocab) + 
                             len(self.original_type_vocab) + 
                             len(self.label_vocab))
                return np.array([]).reshape(0, total_size)
            else:
                return np.array([]).reshape(0, 3)
        
        if use_one_hot:
            return self._encode_one_hot(ipag_nodes)
        else:
            return self._encode_indices(ipag_nodes)
    
    def encode_batch(self, ipag_nodes_batch, use_one_hot=False):
        """
        Encode multiple graphs' nodes.
        
        Args:
            ipag_nodes_batch: List of List of node dicts
            use_one_hot (bool): Encoding type
        
        Returns:
            list: List of numpy arrays, one per graph
        """
        return [self.encode(nodes, use_one_hot) for nodes in ipag_nodes_batch]
    
    def _encode_indices(self, nodes):
        """Encode as indices [num_nodes, 3]."""
        features = np.zeros((len(nodes), 3), dtype=np.int32)
        
        for i, node in enumerate(nodes):
            features[i, 0] = self.node_type_vocab.get(
                node['type'], 
                self.node_type_vocab['<UNK>']
            )
            features[i, 1] = self.original_type_vocab.get(
                node['original_type'], 
                self.original_type_vocab['<UNK>']
            )
            
            label = node.get('label', '')
            if not label or not label.strip():
                features[i, 2] = self.label_vocab['<EMPTY>']
            else:
                features[i, 2] = self.label_vocab.get(
                    label, 
                    self.label_vocab['<UNK>']
                )
        
        return features
    
    def _encode_one_hot(self, nodes):
        """Encode as one-hot vectors."""
        n_type = len(self.node_type_vocab)
        n_orig = len(self.original_type_vocab)
        n_label = len(self.label_vocab)
        total = n_type + n_orig + n_label
        
        features = np.zeros((len(nodes), total), dtype=np.float32)
        
        for i, node in enumerate(nodes):
            # Node type
            idx = self.node_type_vocab.get(node['type'], self.node_type_vocab['<UNK>'])
            features[i, idx] = 1.0
            
            # Original type
            idx = self.original_type_vocab.get(node['original_type'], self.original_type_vocab['<UNK>'])
            features[i, n_type + idx] = 1.0
            
            # Label
            label = node.get('label', '')
            if not label or not label.strip():
                idx = self.label_vocab['<EMPTY>']
            else:
                idx = self.label_vocab.get(label, self.label_vocab['<UNK>'])
            features[i, n_type + n_orig + idx] = 1.0
        
        return features
    
    def get_feature_dim(self, use_one_hot=False):
        """Get feature dimension."""
        if use_one_hot:
            return (len(self.node_type_vocab) + 
                   len(self.original_type_vocab) + 
                   len(self.label_vocab))
        return 3
    
    def get_vocab_sizes(self):
        """Get vocabulary sizes for embedding initialization."""
        return {
            'node_type': len(self.node_type_vocab),
            'original_type': len(self.original_type_vocab),
            'label': len(self.label_vocab),
        }
