import os
import pandas as pd
from joblib import dump
from ipag_gin.graph.ipag_builder import IPAGBuilder
from ipag_gin.graph.build_language import LanguageBuilder
from ipag_gin.utils.gather_processed_cwes import get_processed_cweids

# Root directory where CWE folders are located
root_dir = "/mnt/vstor/courses/csds447/sxi219/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/data/graphs"
chunk_size = 2000

processed_cwe = get_processed_cweids(root_dir)

for cwe in os.listdir(root_dir):
    if cwe not in processed_cwe:
        cwe_folder = os.path.join(root_dir, cwe)
        if not os.path.isdir(cwe_folder):
            continue
        print(f"Processing CWE: {cwe}")
        for cve in os.listdir(cwe_folder):
            cve_folder = os.path.join(cwe_folder, cve)
            if not os.path.isdir(cve_folder):
                continue
            for label in ['0', '1']:
                label_folder = os.path.join(cve_folder, label)
                if not os.path.isdir(label_folder):
                    continue
                csv_path = os.path.join(label_folder, 'data.csv')
                if not os.path.exists(csv_path):
                    continue

                try:
                    reader = pd.read_csv(csv_path, chunksize=chunk_size)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
                    continue

                for i, chunk in enumerate(reader):
                    print(f"Processing chunk {i} in {cwe}/{cve}/{label}...")

                    # Make sure code and language columns are present and well-typed
                    if 'code' not in chunk.columns or 'lang' not in chunk.columns:
                        print(f"Missing 'code'/'lang' columns in {csv_path}, skipping chunk.")
                        continue

                    df_code = chunk['code'].astype(str)
                    df_lang = chunk['lang'].str.lower()

                    # Remove NaN or blank code
                    valid_mask = df_code.notnull() & (df_code.str.strip() != "") & df_lang.notnull()
                    df_code = df_code[valid_mask]
                    df_lang = df_lang[valid_mask]
                    if len(df_code) == 0 or len(df_lang) == 0:
                        print(f"No valid code/lang samples in chunk {i} of {csv_path}")
                        continue

                    # Build language map for all present languages in chunk
                    langs = set(df_lang.unique())
                    lang_map = LanguageBuilder(langs)
                    ipag = IPAGBuilder(source=df_code, language=df_lang, lang_map=lang_map.build())
                    ipag.build()
                    df_ipag = ipag.get_ipag_dataframe()

                    all_ipag_nodes = df_ipag['ipag_nodes'].tolist()
                    all_ipag_edges = df_ipag['ipag_edges'].tolist()

                    # Save in the same label_folder as data.csv
                    out_prefix = os.path.join(label_folder, f"ipag_nodes_chunk_{i}")
                    dump(all_ipag_nodes, f"{out_prefix}.joblib", compress=3)
                    dump(all_ipag_edges, f"{out_prefix.replace('nodes', 'edges')}.joblib", compress=3)
                    print(f"Saved chunk {i} in {label_folder}")

print("Per-CWE preprocessing complete.")
