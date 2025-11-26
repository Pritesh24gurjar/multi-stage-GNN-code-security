import pandas as pd
import torch
from multiprocessing import Process
from ipag_gin.graph.ipag_builder import IPAGBuilder
from ipag_gin.graph.build_language import LanguageBuilder
from ipag_gin.graph.features_accelerated import BuildNodeFeaturesAccelerated
from multiprocessing import Process
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2560)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    data_abs_path = "/mnt/vstor/courses/csds447/sxi219/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/data/processed/MSR_data_cleaned.csv"
    df = pd.read_csv(data_abs_path, nrows=1000)
    df_code_after = df['func_after']
    df_code_lang = df['lang'].str.lower()
    langs = set(df_code_lang.str.lower())
    lang_map = LanguageBuilder(langs)
    ipag = IPAGBuilder(source = df_code_after, language = df_code_lang.str.lower(), lang_map=lang_map.build())
    ipag.build()
    df_ipag = ipag.get_ipag_dataframe()
    all_ipag_nodes = df_ipag['ipag_nodes'].tolist()
    all_ipag_edges = df_ipag['ipag_edges'].tolist()

    mid = len(all_ipag_nodes) // 2
    if args.chunk_idx == 0:
        ipag_nodes, ipag_edges = all_ipag_nodes[:mid], all_ipag_edges[:mid]
        cuda_device_idx = 0
    else:
        ipag_nodes, ipag_edges = all_ipag_nodes[mid:], all_ipag_edges[mid:]
        cuda_device_idx = 1

    builder = BuildNodeFeaturesAccelerated(cuda_device_idx=cuda_device_idx, device='cuda', batch_size=args.batch_size)
    results = builder.process_ipag_batch(ipag_nodes, ipag_edges)
    builder.save_features(results, args.save_path)
