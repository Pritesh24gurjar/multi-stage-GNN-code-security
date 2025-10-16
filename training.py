import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from data_gathering import load_raw_data
from preprocessing import parse_graph_representation, create_graph_data_list
from model import GCN

def train_model(model, train_loader, test_loader, num_epochs=30, learning_rate=0.001):
    """
    Trains and evaluates the GNN model.
    """
    criterion = nn.BCEWithLogitsLoss() # Use BCEWithLogitsLoss for sigmoid output
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting model training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.unsqueeze(1)) # Ensure target has a trailing dimension of 1
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Testing phase
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                out = model(data)
                loss = criterion(out, data.y.unsqueeze(1))
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
    print("Training complete.")

def evaluate_model(model, test_loader):
    """
    Evaluates the trained model on the test dataset and prints accuracy.
    """
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for data in test_loader:
            out = model(data)
            predictions = (out > 0.5).float() # Apply threshold for binary prediction
            correct_predictions += (predictions == data.y.unsqueeze(1)).sum().item()
            total_samples += data.y.size(0)

    accuracy = correct_predictions / total_samples
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

def make_predictions(model, test_dataset, num_samples=5):
    """
    Makes predictions on a few samples from the test dataset.
    """
    selected_samples = test_dataset[:num_samples]
    model.eval()
    print(f"\nPredictions on selected {num_samples} test samples:")
    with torch.no_grad():
        for i, data in enumerate(selected_samples):
            out = model(data)
            prediction = (out > 0.5).float().item() # Apply threshold and get scalar value
            actual_label = data.y.item() # Get scalar value of the actual label
            print(f"Sample {i+1}: Predicted Label = {int(prediction)}, Actual Label = {int(actual_label)}")

if __name__ == "__main__":
    # Full pipeline execution for demonstration
    local_dataset_path = 'dataset/CWE-77/5result/'
    # 1. Data Gathering
    df_raw = load_raw_data(local_dataset_path)
    if df_raw is None or df_raw.empty:
        print("Failed to gather raw data. Exiting.")
        exit()

    # 2. Preprocessing
    df_raw[['edges', 'node_features']] = df_raw['contents'].apply(lambda x: pd.Series(parse_graph_representation(x)))
    graph_data_list = create_graph_data_list(df_raw)

    # Split dataset
    train_dataset, test_dataset = train_test_split(graph_data_list, test_size=0.2, random_state=42)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. Model Definition
    model = GCN(hidden_channels=64)

    # 4. Training
    train_model(model, train_loader, test_loader, num_epochs=30)

    # 5. Evaluation
    accuracy = evaluate_model(model, test_loader)

    # 6. Predictions
    make_predictions(model, test_dataset)