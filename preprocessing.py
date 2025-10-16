import re
import pandas as pd
import torch
from torch_geometric.data import Data

def parse_graph_representation(contents):
    """
    Parses the raw content string to extract edges and raw node features.
    Expected delimiters: "-----children-----", "-----attribute-----", "-----ast_node-----", "-----joern-----".
    """
    edges = []
    node_features_raw = []
    current_section = None

    if contents:
        lines = contents.splitlines()
        for line in lines:
            line = line.strip()
            if line == "-----children-----":
                current_section = "children"
                continue
            elif line == "-----nextToken-----":
                current_section = "nextToken"
                continue
            elif line == "-----computeFrom-----":
                current_section = "computeFrom"
                continue
            elif line == "-----guardedBy-----":
                current_section = "guardedBy"
                continue
            elif line == "-----guardedByNegation-----":
                current_section = "guardedByNegation"
                continue
            elif line == "-----lastLexicalUse-----":
                current_section = "lastLexicalUse"
                continue
            elif line == "-----jump-----":
                current_section = "jump"
                continue
            elif line == "-----attribute-----":
                current_section = "attribute"
                continue
            elif line == "-----ast_node-----":
                current_section = "ast_node"
                continue
            elif line == "-----joern-----":
                current_section = "joern"
                continue
            elif line.startswith("-----"): # Handle other potential delimiters
                current_section = None
                continue

            if current_section == "children":
                match = re.match(r'(\d+),(\d+)', line)
                if match:
                    edges.append((int(match.group(1)), int(match.group(2))))
            elif current_section in ["attribute", "ast_node", "joern"]:
                if line: # Avoid adding empty lines
                    node_features_raw.append(line)
    
    return edges, node_features_raw

def create_graph_data_list(df):
    """
    Converts a DataFrame with 'edges', 'node_features', and 'label' columns
    into a list of PyTorch Geometric Data objects.
    """
    graph_data_list = []
    print("Creating graph data objects...")
    for index, row in df.iterrows():
        edges = row['edges']
        node_features_raw = row['node_features']
        label = row['label']

        if not edges:
            continue # Skip if there are no edges

        # Create a mapping from original node indices to continuous indices
        unique_nodes = sorted(list(set([node for edge in edges for node in edge])))
        node_mapping = {old_index: new_index for new_index, old_index in enumerate(unique_nodes)}

        # Re-index edges
        reindexed_edges = [(node_mapping[u], node_mapping[v]) for u, v in edges]
        edge_index = torch.tensor(reindexed_edges, dtype=torch.long).t().contiguous()

        # Process node features (basic approach: use raw strings as features for now)
        # This part will need refinement for a real GNN
        num_nodes = len(unique_nodes)
        node_features = torch.ones((num_nodes, 1), dtype=torch.float) # Placeholder

        # Create PyTorch Geometric Data object
        data = Data(edge_index=edge_index, x=node_features, y=torch.tensor([label], dtype=torch.float))
        graph_data_list.append(data)
    
    print(f"Created {len(graph_data_list)} graph data objects.")
    return graph_data_list

