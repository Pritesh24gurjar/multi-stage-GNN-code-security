import os
import zipfile
import pandas as pd

def load_raw_data(directory_path):
    """
    Loads raw data from individual files in a directory into a pandas DataFrame.
    Each file is expected to contain a label and content separated by markers.
    """
    data_list = []
    print(f"Loading data from: {directory_path}")
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                lines = content.splitlines()
                
                label = None
                if len(lines) > 1 and "-----label-----" in lines[0]:
                    try:
                        label = int(lines[1])
                    except ValueError:
                        label = None # Handle cases where label is not an integer

                code_start_index = -1
                for i, line in enumerate(lines):
                    if "-----code-----" in line:
                        code_start_index = i
                        break
                
                code_content = "\n".join(lines[code_start_index + 1:]) if code_start_index != -1 else content

                data_list.append({'filename': filename, 'label': label, 'contents': code_content})
    
    df = pd.DataFrame(data_list)
    print(f"Loaded {len(df)} entries.")
    return df

if __name__ == "__main__":
    
    local_dataset_path = 'dataset/CWE-77/5result/'
    
    # Load data
    if os.path.exists(local_dataset_path):
        df_raw = load_raw_data(local_dataset_path)
        print("Raw Data Head:")
        print(df_raw.head())
        print("\nRaw Data Info:")
        df_raw.info()
    else:
        print(f"Local dataset not found at: {local_dataset_path}. Please ensure the dataset is in the correct location.")