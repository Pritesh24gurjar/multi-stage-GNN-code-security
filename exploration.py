import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from data_gathering import load_raw_data
from preprocessing import parse_graph_representation

def perform_eda(df):
    """
    Performs exploratory data analysis on the loaded DataFrame.
    """
    print("Performing Exploratory Data Analysis...")
    
    print("\nDataFrame Info:")
    df.info()
    
    print("\nFirst 5 rows of the DataFrame:")
    print(df.head())
    
    print("\nValue counts for 'label' column:")
    print(df['label'].value_counts())
    
    # Plotting label distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=df)
    plt.title('Distribution of Labels (Vulnerable vs. Non-Vulnerable)')
    plt.xlabel('Label (0: Non-Vulnerable, 1: Vulnerable)')
    plt.ylabel('Number of Samples')
    plt.show()
    
    # Analyze number of edges and nodes
    df['num_edges'] = df['edges'].apply(len)
    df['num_nodes'] = df['node_features'].apply(lambda x: len(set([node for edge in x for node in edge])) if isinstance(x, list) else 0) # Corrected for node_features being raw strings
    
    print("\nDescriptive statistics for number of edges:")
    print(df['num_edges'].describe())
    
    print("\nDescriptive statistics for number of nodes:")
    print(df['num_nodes'].describe())
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['num_edges'], bins=30, kde=True)
    plt.title('Distribution of Number of Edges')
    plt.xlabel('Number of Edges')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df['num_nodes'], bins=30, kde=True)
    plt.title('Distribution of Number of Nodes')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # Analyze common node features (if applicable and after proper tokenization/vectorization)
    # For now, let's just look at the raw node feature strings if they exist
    all_node_features = [item for sublist in df['node_features'] if isinstance(sublist, list) for item in sublist]
    if all_node_features:
        feature_counts = Counter(all_node_features)
        print("\nMost common raw node features:")
        for feature, count in feature_counts.most_common(10):
            print(f"- {feature}: {count}")
    else:
        print("\nNo raw node features found for analysis.")

if __name__ == "__main__":
    # Full pipeline execution for demonstration
    local_dataset_path = 'dataset/CWE-77/5result/'
    
    # 1. Data Gathering
    df_raw = load_raw_data(local_dataset_path)
    
    if df_raw is None or df_raw.empty:
        print("Failed to gather raw data. Exiting.")
        exit()

    # 2. Preprocessing for EDA (only graph parsing needed)
    df_raw[['edges', 'node_features']] = df_raw['contents'].apply(lambda x: pd.Series(parse_graph_representation(x)))
    
    # 3. Perform EDA
    perform_eda(df_raw)