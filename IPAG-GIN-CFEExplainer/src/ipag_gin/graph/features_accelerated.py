from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
from collections import Counter
import pickle


class BuildNodeFeaturesAccelerated:
    """
    Extract features and embeddings from IPAG nodes and edges using GraphCodeBERT
    utilizing GPU parallelization and true batch processing across multiple graphs.
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


    def encode_text_batch(self, texts, max_length=64):
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
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings

    def get_all_node_embeddings_batch(self, ipag_nodes_list):
        """
        Processes all nodes from all IPAGs in a single, large batched run.

        Args:
            ipag_nodes_list (list): A list where each item is an IPAG's node list.

        Returns:
            dict: A map where Key=(graph_index, node_id), Value=embedding
        """
        all_texts = []
        # Stores (graph_index, node_id) to map embeddings back
        all_node_identifiers = []

        print(f"Collecting texts from {len(ipag_nodes_list)} graphs...")

        # === 1. Collect all node texts from ALL graphs ===
        for graph_idx, nodes in enumerate(ipag_nodes_list):
            for node in nodes:
                # Logic copied from your original get_node_embeddings
                node_id = node['id']
                label = node.get('label', '')
                node_type = node.get('type', 'PROPERTY')
                original_type = node.get('original_type', '')

                if label:
                    text = f"{node_type}: {label}"
                else:
                    text = f"{node_type}: {original_type}"

                all_texts.append(text)
                all_node_identifiers.append((graph_idx, node_id))

        if not all_texts:
            return {}

        print(f"Collected {len(all_texts)} total nodes. Starting batched encoding...")

        # === 2. Run ONE large batched inference pass ===
        all_embeddings_list = []
        for i in range(0, len(all_texts), self.batch_size):
            batch_texts = all_texts[i:i + self.batch_size]

            # This is your existing, efficient batch encoding function
            embeddings = self.encode_text_batch(batch_texts)
            all_embeddings_list.append(embeddings)

            # Print progress every 10 batches
            if (i + self.batch_size) % (self.batch_size * 10) == 0:
                 print(f"  ...encoded {min(i + self.batch_size, len(all_texts))}/{len(all_texts)} total nodes...")

        # Combine all batch results into one large array
        all_embeddings = np.concatenate(all_embeddings_list, axis=0)

        # === 3. Map embeddings back to graphs/nodes ===
        global_embedding_map = {} # Key: (graph_idx, node_id), Value: embedding
        for identifier, embedding in zip(all_node_identifiers, all_embeddings):
            global_embedding_map[identifier] = embedding

        print("Global embedding map created.")
        return global_embedding_map


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
        Efficiently compute structural features (degrees) for all nodes.
        """
        in_counts = Counter()
        out_counts = Counter()

        for edge in edges:
            out_counts[edge['source']] += 1
            in_counts[edge['target']] += 1

        features = {}
        for node_id in node_ids:
            in_degree = in_counts.get(node_id, 0)
            out_degree = out_counts.get(node_id, 0)

            features[node_id] = {
                'degree': in_degree + out_degree,
                'in_degree': in_degree,
                'out_degree': out_degree,
                'is_leaf': 1 if out_degree == 0 else 0,
                'is_root': 1 if in_degree == 0 else 0
            }

        return features

    def get_node_features_from_precomputed(self, ipag_nodes, ipag_edges, precomputed_node_embeddings):
        """
        Extracts comprehensive features for all nodes, using pre-computed embeddings.

        Args:
            ipag_nodes (list): List of node dictionaries
            ipag_edges (list): List of edge dictionaries
            precomputed_node_embeddings (dict): A dict mapping {node_id: embedding}

        Returns:
            dict: Dictionary mapping node_id to feature dictionary
        """
        if not ipag_nodes:
            return {}

        # === Use the provided embeddings ===
        node_embeddings = precomputed_node_embeddings

        # Get type encodings in batch (this is already fast)
        node_ids = [node['id'] for node in ipag_nodes]
        node_types = [node.get('type', 'PROPERTY') for node in ipag_nodes]
        type_encodings = self._get_node_type_encoding_batch(node_types)

        # Get structural features in bulk (this is already fast)
        structural_features = self._compute_structural_features_bulk(node_ids, ipag_edges)

        # Combine all features
        node_features = {}
        feature_dim = 0
        default_embedding = np.zeros(768, dtype=np.float32) # Assumes 768 hidden size

        for i, node in enumerate(ipag_nodes):
            node_id = node['id']
            node_type = node_types[i]

            # Use default embedding if one is missing
            embedding = node_embeddings.get(node_id, default_embedding)
            type_encoding = type_encodings[i]
            structural = structural_features.get(node_id, {
                'degree': 0, 'in_degree': 0, 'out_degree': 0, 'is_leaf': 0, 'is_root': 0
            })

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
            feature_dim = len(combined)

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

        if feature_dim > 0 and len(ipag_nodes) > 10: # Avoid spamming for tiny graphs
            print(f"Feature combination complete. Feature dimension: {feature_dim}")
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

        # Add epsilon for numerical stability
        norms_product = (norms_source.squeeze() * norms_target.squeeze()) + 1e-8
        dot_product = np.sum(source_embeddings * target_embeddings, axis=1)
        cosine_sim = dot_product / norms_product

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
        if node_features:
            all_embeddings = np.array([feat['embedding'] for feat in node_features.values()])
            avg_embedding = np.mean(all_embeddings, axis=0)

            # Vectorized structural statistics
            degrees = np.array([feat['metadata']['degree'] for feat in node_features.values()])
            avg_degree = np.mean(degrees)
            max_degree = np.max(degrees)

            num_leaves = sum(1 for feat in node_features.values() if feat['metadata']['is_leaf'])
            num_roots = sum(1 for feat in node_features.values() if feat['metadata']['is_root'])
        else:
            avg_embedding = np.zeros(768) # Default embedding size
            avg_degree = 0
            max_degree = 0
            num_leaves = 0
            num_roots = 0


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
        Process multiple IPAGs in batch using the accelerated workflow.

        Args:
            ipag_nodes_list (list): List of IPAG node lists
            ipag_edges_list (list): List of IPAG edge lists

        Returns:
            list: List of feature dictionaries, one per IPAG
        """
        results = []

        # === STAGE 1: Get ALL embeddings for ALL graphs in one go ===
        global_embedding_map = self.get_all_node_embeddings_batch(ipag_nodes_list)

        default_embedding = np.zeros(768, dtype=np.float32)

        # === STAGE 2: Process each graph's other features (now very fast) ===
        for idx, (nodes, edges) in enumerate(zip(ipag_nodes_list, ipag_edges_list)):

            if (idx + 1) % 10 == 0 or idx == 0 or idx == len(ipag_nodes_list) - 1:
                print(f"\nProcessing IPAG {idx + 1}/{len(ipag_nodes_list)} (fast features)...")

            # Create a local embedding map for just this graph
            local_node_embeddings = {
                node['id']: global_embedding_map.get((idx, node['id']), default_embedding)
                for node in nodes
            }

            # Use the new function that accepts pre-computed embeddings
            node_features = self.get_node_features_from_precomputed(nodes, edges, local_node_embeddings)

            # These functions are already fast and can be called as-is
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
            pickle.dump(features, f, protocol=4)
        print(f"Features saved to {filepath}")

    def load_features(self, filepath):
        """Load features from disk."""
        with open(filepath, 'rb') as f:
            features = pickle.load(f)
        print(f"Features loaded from {filepath}")
        return features