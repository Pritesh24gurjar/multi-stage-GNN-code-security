import argparse
from ipag_gin.graph.features_accelerated import BuildNodeFeaturesAccelerated
from joblib import load
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes_path", type=str, required=True)
    parser.add_argument("--edges_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--cuda_device_idx", type=int, default=0)
    args = parser.parse_args()

    ipag_nodes = load(args.nodes_path)
    ipag_edges = load(args.edges_path)

    builder = BuildNodeFeaturesAccelerated(
        cuda_device_idx=args.cuda_device_idx,
        device='cuda',
        batch_size=args.batch_size
    )
    results = builder.process_ipag_batch(ipag_nodes, ipag_edges)
    builder.save_features(results, args.save_path)
