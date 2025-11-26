from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
from collections import Counter
import pickle

class BuildNodeFeatures:
    """
    Extract features and embeddings from IPAG nodes and edges using GraphCodeBERT.
    """
    
    def __init__(self, device=None):
        """
        Initialize the feature builder with GraphCodeBERT model.
        
        Args:
            device (str): Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
        self.model.eval()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        print(f"BuildNodeFeatures initialized on device: {self.device}")
    
    def _encode_text(self, text, max_length=512):
        """
        Encode text using GraphCodeBERT tokenizer and get embeddings.
        
        Args:
            text (str): Text to encode
            max_length (int): Maximum sequence length
            
        Returns:
            torch.Tensor: Embeddings of shape (hidden_size,)
        """
        if not text or not isinstance(text, str):
            text = ""
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding as representation
            embeddings = outputs.last_hidden_state[:, 0, :].squeeze(0)
        
        return embeddings
    
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
    
    def _compute_structural_features(self, node_id, edges):
        """
        Compute structural features for a node based on graph connectivity.
        
        Args:
            node_id (str): ID of the node
            edges (list): List of edge dictionaries with 'source' and 'target'
            
        Returns:
            dict: Structural features including degree, in-degree, out-degree
        """
        in_degree = 0
        out_degree = 0
        
        for edge in edges:
            if edge['source'] == node_id:
                out_degree += 1
            if edge['target'] == node_id:
                in_degree += 1
        
        degree = in_degree + out_degree
        
        return {
            'degree': degree,
            'in_degree': in_degree,
            'out_degree': out_degree,
            'is_leaf': 1 if out_degree == 0 else 0,
            'is_root': 1 if in_degree == 0 else 0
        }
    
    def get_node_embeddings(self, ipag_nodes, batch_size=32):
        """
        Get GraphCodeBERT embeddings for all nodes in an IPAG.
        
        Args:
            ipag_nodes (list): List of node dictionaries with 'id', 'type', 'label'
            batch_size (int): Batch size for processing
            
        Returns:
            dict: Dictionary mapping node_id to embedding tensor
        """
        if not ipag_nodes:
            return {}
        
        node_embeddings = {}
        
        # Process in batches
        for i in range(0, len(ipag_nodes), batch_size):
            batch_nodes = ipag_nodes[i:i + batch_size]
            
            for node in batch_nodes:
                node_id = node['id']
                label = node.get('label', '')
                node_type = node.get('type', 'PROPERTY')
                original_type = node.get('original_type', '')
                
                # Combine label and type information for encoding
                if label:
                    text = f"{node_type}: {label}"
                else:
                    text = f"{node_type}: {original_type}"
                
                # Get embedding
                embedding = self._encode_text(text)
                node_embeddings[node_id] = embedding.cpu().numpy()
            
            if (i + batch_size) % 100 == 0:
                print(f"Processed {min(i + batch_size, len(ipag_nodes))}/{len(ipag_nodes)} nodes...")
        
        return node_embeddings
    
    def get_node_features(self, ipag_nodes, ipag_edges):
        """
        Extract comprehensive features for all nodes in an IPAG.
        
        Args:
            ipag_nodes (list): List of node dictionaries
            ipag_edges (list): List of edge dictionaries
            
        Returns:
            dict: Dictionary mapping node_id to feature dictionary containing:
                - 'embedding': GraphCodeBERT embedding (768-dim)
                - 'type_encoding': One-hot node type (3-dim)
                - 'structural': Structural graph features (5-dim)
                - 'combined': Concatenated feature vector (776-dim)
        """
        if not ipag_nodes:
            return {}
        
        print(f"Extracting features for {len(ipag_nodes)} nodes...")
        
        # Get embeddings
        node_embeddings = self.get_node_embeddings(ipag_nodes)
        
        # Compute features for each node
        node_features = {}
        
        for node in ipag_nodes:
            node_id = node['id']
            node_type = node.get('type', 'PROPERTY')
            
            # Get embedding
            embedding = node_embeddings.get(node_id, np.zeros(768, dtype=np.float32))
            
            # Get type encoding
            type_encoding = self._get_node_type_encoding(node_type)
            
            # Get structural features
            structural = self._compute_structural_features(node_id, ipag_edges)
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
        Extract features for edges based on connected nodes.
        
        Args:
            ipag_edges (list): List of edge dictionaries with 'source', 'target', 'type'
            node_features (dict): Dictionary of node features from get_node_features()
            
        Returns:
            list: List of edge feature dictionaries
        """
        if not ipag_edges:
            return []
        
        edge_features = []
        
        for edge in ipag_edges:
            source_id = edge['source']
            target_id = edge['target']
            edge_type = edge.get('type', 'CHILD')
            
            # Get node features
            source_feat = node_features.get(source_id, {})
            target_feat = node_features.get(target_id, {})
            
            if not source_feat or not target_feat:
                continue
            
            # Combine source and target embeddings
            source_emb = source_feat.get('embedding', np.zeros(768))
            target_emb = target_feat.get('embedding', np.zeros(768))
            
            # Create edge feature vector
            edge_feature = {
                'source': source_id,
                'target': target_id,
                'type': edge_type,
                'source_embedding': source_emb,
                'target_embedding': target_emb,
                'concatenated': np.concatenate([source_emb, target_emb]),
                'element_wise_product': source_emb * target_emb,
                'cosine_similarity': self._cosine_similarity(source_emb, target_emb)
            }
            
            edge_features.append(edge_feature)
        
        return edge_features
    
    def _cosine_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1 (np.ndarray): First vector
            vec2 (np.ndarray): Second vector
            
        Returns:
            float: Cosine similarity
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
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
        
        # Count statistics
        num_nodes = len(ipag_nodes)
        num_edges = len(ipag_edges)
        
        node_type_counts = Counter(node.get('type') for node in ipag_nodes)
        
        # Compute average embeddings
        all_embeddings = [feat['embedding'] for feat in node_features.values()]
        avg_embedding = np.mean(all_embeddings, axis=0) if all_embeddings else np.zeros(768)
        
        # Compute structural statistics
        degrees = [feat['metadata']['degree'] for feat in node_features.values()]
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        
        num_leaves = sum(1 for feat in node_features.values() if feat['metadata']['is_leaf'])
        num_roots = sum(1 for feat in node_features.values() if feat['metadata']['is_root'])
        
        graph_features = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_tokens': node_type_counts.get('TOKEN', 0),
            'num_declarations': node_type_counts.get('DECLARATION', 0),
            'num_properties': node_type_counts.get('PROPERTY', 0),
            'avg_degree': avg_degree,
            'max_degree': max_degree,
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
            
            # Get node features
            node_features = self.get_node_features(nodes, edges)
            
            # Get edge features
            edge_features = self.get_edge_features(edges, node_features)
            
            # Get graph-level features
            graph_features = self.get_graph_level_features(nodes, edges, node_features)
            
            results.append({
                'node_features': node_features,
                'edge_features': edge_features,
                'graph_features': graph_features
            })
        
        print(f"\nBatch processing complete. Processed {len(results)} IPAGs.")
        return results
    
    def save_features(self, features, filepath):
        """
        Save extracted features to disk.
        
        Args:
            features (dict or list): Features to save
            filepath (str): Path to save file
        """
        
        
        with open(filepath, 'wb') as f:
            pickle.dump(features, f)
        
        print(f"Features saved to {filepath}")
    
    def load_features(self, filepath):
        """
        Load features from disk.
        
        Args:
            filepath (str): Path to load file
            
        Returns:
            Features dictionary
        """
        
        
        with open(filepath, 'rb') as f:
            features = pickle.load(f)
        
        print(f"Features loaded from {filepath}")
        return features