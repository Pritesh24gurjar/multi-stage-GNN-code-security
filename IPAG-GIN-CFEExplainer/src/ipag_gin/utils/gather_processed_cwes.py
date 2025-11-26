import os

def get_processed_cweids(root_folder, feature_keyword="feature"):
    processed_cwes = []
    for cwe_id in os.listdir(root_folder):
        cwe_path = os.path.join(root_folder, cwe_id)
        if not os.path.isdir(cwe_path):
            continue
        complete = True
        for cve_id in os.listdir(cwe_path):
            cve_path = os.path.join(cwe_path, cve_id)
            if not os.path.isdir(cve_path):
                continue
            for label in ["0", "1"]:
                label_path = os.path.join(cve_path, label)
                if not os.path.isdir(label_path):
                    complete = False
                    break
                files = os.listdir(label_path)
                num_feats = len([f for f in files if feature_keyword in f])
                expected = len(files) // 3
                if num_feats != expected:
                    complete = False
                    break
            if not complete:
                break
        if complete:
            processed_cwes.append(cwe_id)
    return processed_cwes

if __name__ == "__main__":
    root = sys.argv[1]
    output = sys.argv[2]
    ids = set(get_processed_cweids(root))
    # Read old ids if output file exists
    if os.path.exists(output):
        with open(output, "r") as f:
            old_ids = set(line.strip() for line in f if line.strip())
    else:
        old_ids = set()

    new_ids = ids - old_ids

    if new_ids:
        with open(output, "a") as f:
            for i in sorted(new_ids):
                f.write(f"{i}\n")
        print(f"Appended {len(new_ids)} new processed CWEIDs to {output}")
    else:
        print("No new CWEIDs to append.")
