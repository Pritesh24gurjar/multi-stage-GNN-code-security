from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
from collections import Counter
import pickle
import graph_structure_features as struct_features  # C++ extension


class BuildNodeFeaturesAccelerated:
    """
    Extract features and embeddings from IPAG nodes and edges using GraphCodeBERT utilizing GPU parallelization.
    """
    
    def __init__(self, device=None, cuda_device_idx=0, batch_size=128):
        """
        Initialize the feature builder with GraphCodeBERT model.

        Args:
            device (str or None): Torch device string; e.g., 'cuda', 'cpu'.
            cuda_device_idx (int): CUDA device index for multi-GPU systems.
            batch_size (int): Batch size to use for efficient inference.
        """
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
        self.model.eval()
        
        # Device selection and assignment
        if device is None:
            # Use specific GPU index if available, else fallback to CPU
            if torch.cuda.is_available():
                self.device = torch.device(f'cuda:{cuda_device_idx}')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.batch_size = batch_size
        
        print(f"BuildNodeFeaturesAccelerated initialized on device: {self.device} with batch size: {self.batch_size}")
        print(f"C++ acceleration: {'ENABLED' if CPP_AVAILABLE else 'DISABLED'}")
    
    def encode_text_batch(self, texts, max_length=512):
        """
        Encode a list of texts in a single batch using GraphCodeBERT.
        
        Args:
            texts (List[str]): List of node texts to encode.
            max_length (int): Maximum sequence length.
        Returns:
            np.ndarray: Embedding array (batch_size, hidden_size)
        """
        sanitized = [t if isinstance(t, str) else "" for t in texts]
        inputs = self.tokenizer(
            sanitized,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # (batch_size, hidden_size)
        return embeddings

    def get_node_embeddings(self, ipag_nodes):
        """
        Get GraphCodeBERT embeddings for all nodes in an IPAG using batch processing.
        
        Args:
            ipag_nodes (list): List of node dictionaries with 'id', 'type', 'label'
            
        Returns:
            dict: Dictionary mapping node_id to embedding array
        """
        if not ipag_nodes:
            return {}
        
        node_embeddings = {}
        node_ids = []
        texts = []
        
        # Prepare all texts first
        for node in ipag_nodes:
            node_id = node['id']
            label = node.get('label', '')
            node_type = node.get('type', 'PROPERTY')
            original_type = node.get('original_type', '')
            
            # Combine label and type information for encoding
            if label:
                text = f"{node_type}: {label}"
            else:
                text = f"{node_type}: {original_type}"
            
            node_ids.append(node_id)
            texts.append(text)
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_ids = node_ids[i:i + self.batch_size]
            
            # Get embeddings for batch
            embeddings = self.encode_text_batch(batch_texts)
            
            # Store embeddings
            for node_id, embedding in zip(batch_ids, embeddings):
                node_embeddings[node_id] = embedding
            
            if (i + self.batch_size) % 500 == 0:
                print(f"Processed {min(i + self.batch_size, len(texts))}/{len(texts)} nodes...")
        
        return node_embeddings
    
    def _get_node_type_encoding(self, node_type):
        """
        Create one-hot encoding for node type.
        
        Args:
            node_type (str): Node type ('TOKEN', 'DECLARATION', 'PROPERTY')
            
        Returns:
            np.ndarray: One-hot encoded vector
        """
        type_map = {'TOKEN': 0, 'DECLARATION': 1, 'PROPERTY': 2}
        encoding = np.zeros(3, dtype=np.float32)
        
        if node_type in type_map:
            encoding[type_map[node_type]] = 1.0
        else:
            # Default to PROPERTY if unknown
            encoding[2] = 1.0
        
        return encoding
    
    def _get_node_type_encoding_batch(self, node_types):
        """
        Create one-hot encodings for multiple node types at once.
        
        Args:
            node_types (list): List of node type strings
            
        Returns:
            np.ndarray: Array of one-hot encodings (n_nodes, 3)
        """
        type_map = {'TOKEN': 0, 'DECLARATION': 1, 'PROPERTY': 2}
        encodings = np.zeros((len(node_types), 3), dtype=np.float32)
        
        for i, node_type in enumerate(node_types):
            if node_type in type_map:
                encodings[i, type_map[node_type]] = 1.0
            else:
                encodings[i, 2] = 1.0  # Default to PROPERTY
        
        return encodings
    
    def _compute_structural_features_bulk(self, node_ids, edges):
        """
        Compute structural features for all nodes using C++ acceleration if available.
        
        Args:
            node_ids (list): List of node IDs
            edges (list): List of edge dictionaries with 'source' and 'target'
            
        Returns:
            dict: Dictionary mapping node_id to structural features
        """
        if CPP_AVAILABLE:
            # Use C++ implementation
            edge_pairs = [(e['source'], e['target']) for e in edges]
            cpp_features = struct_features.compute_structural_features_bulk(node_ids, edge_pairs)
            
            # Convert to dictionary format
            features = {}
            for node_id in node_ids:
                feat = cpp_features[node_id]
                features[node_id] = {
                    'degree': feat.degree,
                    'in_degree': feat.in_degree,
                    'out_degree': feat.out_degree,
                    'is_leaf': 1 if feat.is_leaf else 0,
                    'is_root': 1 if feat.is_root else 0
                }
            return features
        else:
            # Fallback to Python implementation
            return self._compute_structural_features_bulk_python(node_ids, edges)
    
    def _compute_structural_features_bulk_python(self, node_ids, edges):
        """
        Python fallback for structural feature computation.
        """
        in_counts = {}
        out_counts = {}
        
        for edge in edges:
            source = edge['source']
            target = edge['target']
            out_counts[source] = out_counts.get(source, 0) + 1
            in_counts[target] = in_counts.get(target, 0) + 1
        
        features = {}
        for node_id in node_ids:
            in_degree = in_counts.get(node_id, 0)
            out_degree = out_counts.get(node_id, 0)
            degree = in_degree + out_degree
            
            features[node_id] = {
                'degree': degree,
                'in_degree': in_degree,
                'out_degree': out_degree,
                'is_leaf': 1 if out_degree == 0 else 0,
                'is_root': 1 if in_degree == 0 else 0
            }
        
        return features
    
    def get_node_features(self, ipag_nodes, ipag_edges):
        """
        Extract comprehensive features for all nodes in an IPAG using vectorized operations.
        
        Args:
            ipag_nodes (list): List of node dictionaries
            ipag_edges (list): List of edge dictionaries
            
        Returns:
            dict: Dictionary mapping node_id to feature dictionary
        """
        if not ipag_nodes:
            return {}
        
        print(f"Extracting features for {len(ipag_nodes)} nodes...")
        
        # Get embeddings in batch
        node_embeddings = self.get_node_embeddings(ipag_nodes)
        
        # Get type encodings in batch
        node_ids = [node['id'] for node in ipag_nodes]
        node_types = [node.get('type', 'PROPERTY') for node in ipag_nodes]
        type_encodings = self._get_node_type_encoding_batch(node_types)
        
        # Get structural features in bulk
        structural_features = self._compute_structural_features_bulk(node_ids, ipag_edges)
        
        # Combine all features
        node_features = {}
        for i, node in enumerate(ipag_nodes):
            node_id = node['id']
            node_type = node_types[i]
            
            embedding = node_embeddings.get(node_id, np.zeros(768, dtype=np.float32))
            type_encoding = type_encodings[i]
            structural = structural_features[node_id]
            
            structural_vector = np.array([
                structural['degree'],
                structural['in_degree'],
                structural['out_degree'],
                structural['is_leaf'],
                structural['is_root']
            ], dtype=np.float32)
            
            # Combine all features
            combined = np.concatenate([
                embedding,
                type_encoding,
                structural_vector
            ])
            
            node_features[node_id] = {
                'embedding': embedding,
                'type_encoding': type_encoding,
                'structural': structural_vector,
                'combined': combined,
                'metadata': {
                    'type': node_type,
                    'label': node.get('label', ''),
                    'original_type': node.get('original_type', ''),
                    **structural
                }
            }
        
        print(f"Feature extraction complete. Feature dimension: {len(combined)}")
        return node_features
    
    def get_edge_features(self, ipag_edges, node_features):
        """
        Extract features for edges based on connected nodes using vectorized operations.
        
        Args:
            ipag_edges (list): List of edge dictionaries
            node_features (dict): Dictionary of node features
            
        Returns:
            list: List of edge feature dictionaries
        """
        if not ipag_edges:
            return []
        
        edge_features = []
        
        # Collect all source and target embeddings
        source_embeddings = []
        target_embeddings = []
        valid_edges = []
        
        for edge in ipag_edges:
            source_id = edge['source']
            target_id = edge['target']
            
            source_feat = node_features.get(source_id)
            target_feat = node_features.get(target_id)
            
            if source_feat and target_feat:
                source_embeddings.append(source_feat['embedding'])
                target_embeddings.append(target_feat['embedding'])
                valid_edges.append(edge)
        
        if not valid_edges:
            return []
        
        # Convert to numpy arrays for vectorized operations
        source_embeddings = np.array(source_embeddings)
        target_embeddings = np.array(target_embeddings)
        
        # Vectorized computations
        concatenated = np.concatenate([source_embeddings, target_embeddings], axis=1)
        element_wise_product = source_embeddings * target_embeddings
        
        # Vectorized cosine similarity
        norms_source = np.linalg.norm(source_embeddings, axis=1, keepdims=True)
        norms_target = np.linalg.norm(target_embeddings, axis=1, keepdims=True)
        cosine_sim = np.sum(source_embeddings * target_embeddings, axis=1) / (
            (norms_source.squeeze() * norms_target.squeeze()) + 1e-8
        )
        
        # Build edge feature list
        for i, edge in enumerate(valid_edges):
            edge_feature = {
                'source': edge['source'],
                'target': edge['target'],
                'type': edge.get('type', 'CHILD'),
                'source_embedding': source_embeddings[i],
                'target_embedding': target_embeddings[i],
                'concatenated': concatenated[i],
                'element_wise_product': element_wise_product[i],
                'cosine_similarity': cosine_sim[i]
            }
            edge_features.append(edge_feature)
        
        return edge_features
    
    def get_graph_level_features(self, ipag_nodes, ipag_edges, node_features):
        """
        Extract graph-level features for the entire IPAG.
        
        Args:
            ipag_nodes (list): List of node dictionaries
            ipag_edges (list): List of edge dictionaries
            node_features (dict): Dictionary of node features
            
        Returns:
            dict: Graph-level feature dictionary
        """
        if not ipag_nodes:
            return {}
        
        num_nodes = len(ipag_nodes)
        num_edges = len(ipag_edges)
        
        node_type_counts = Counter(node.get('type') for node in ipag_nodes)
        
        # Vectorized computation of average embeddings
        all_embeddings = np.array([feat['embedding'] for feat in node_features.values()])
        avg_embedding = np.mean(all_embeddings, axis=0) if len(all_embeddings) > 0 else np.zeros(768)
        
        # Vectorized structural statistics
        degrees = np.array([feat['metadata']['degree'] for feat in node_features.values()])
        avg_degree = np.mean(degrees) if len(degrees) > 0 else 0
        max_degree = np.max(degrees) if len(degrees) > 0 else 0
        
        num_leaves = sum(1 for feat in node_features.values() if feat['metadata']['is_leaf'])
        num_roots = sum(1 for feat in node_features.values() if feat['metadata']['is_root'])
        
        graph_features = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_tokens': node_type_counts.get('TOKEN', 0),
            'num_declarations': node_type_counts.get('DECLARATION', 0),
            'num_properties': node_type_counts.get('PROPERTY', 0),
            'avg_degree': float(avg_degree),
            'max_degree': int(max_degree),
            'num_leaves': num_leaves,
            'num_roots': num_roots,
            'avg_embedding': avg_embedding,
            'graph_density': num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        }
        
        return graph_features
    
    def process_ipag_batch(self, ipag_nodes_list, ipag_edges_list):
        """
        Process multiple IPAGs in batch.
        
        Args:
            ipag_nodes_list (list): List of IPAG node lists
            ipag_edges_list (list): List of IPAG edge lists
            
        Returns:
            list: List of feature dictionaries, one per IPAG
        """
        results = []
        
        for idx, (nodes, edges) in enumerate(zip(ipag_nodes_list, ipag_edges_list)):
            print(f"\nProcessing IPAG {idx + 1}/{len(ipag_nodes_list)}...")
            
            node_features = self.get_node_features(nodes, edges)
            edge_features = self.get_edge_features(edges, node_features)
            graph_features = self.get_graph_level_features(nodes, edges, node_features)
            
            results.append({
                'node_features': node_features,
                'edge_features': edge_features,
                'graph_features': graph_features
            })
        
        print(f"\nBatch processing complete. Processed {len(results)} IPAGs.")
        return results
    
    def save_features(self, features, filepath):
        """Save extracted features to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(features, f)
        print(f"Features saved to {filepath}")
    
    def load_features(self, filepath):
        """Load features from disk."""
        with open(filepath, 'rb') as f:
            features = pickle.load(f)
        print(f"Features loaded from {filepath}")
        return features