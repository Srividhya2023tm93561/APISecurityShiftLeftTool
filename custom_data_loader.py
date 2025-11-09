import json
import yaml
import os

def load_custom_dataset():
    """Loads a local custom dataset for training ML models on API specs."""
    custom_dataset_path = "custom_dataset_120.json"
    print(f"Loading dataset from: {os.path.abspath(custom_dataset_path)}")

    if not os.path.exists(custom_dataset_path):
        print(f"Error: Custom dataset file '{custom_dataset_path}' not found.")
        print("Please ensure the file exists in the project root.")
        return [], []

    try:
        with open(custom_dataset_path, 'r') as f:
            dataset_meta = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading dataset file: {e}")
        return [], []

    # Handle both possible formats: new (dict) or old (string list)
    api_specs = dataset_meta.get("api_specs", dataset_meta)
    print(f"Total entries found: {len(api_specs)}")
    if api_specs:
        print("🔹 First entry type:", type(api_specs[0]))
        print("🔹 First entry sample:", api_specs[0])

    texts, labels = [], []

    for entry in api_specs:
        # Handle both cases
        if isinstance(entry, dict):
            filename = entry["filename"]
            label = int(entry["vulnerable"])
        elif isinstance(entry, str):  # fallback for older datasets
            filename = entry
            label = 1 if "vulnerable" in filename.lower() else 0
        else:
            print(f"Skipping unrecognized entry type: {type(entry)}")
            continue

        spec_file_path = os.path.join(os.getcwd(), 'dataset', filename)

        if not os.path.exists(spec_file_path):
            print(f"Skipping missing file: {spec_file_path}")
            continue

        try:
            with open(spec_file_path, 'r', encoding='utf-8') as f:
                api_spec = yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as exc:
            print(f"Skipping file {spec_file_path} due to error: {exc}")
            continue

        flattened_text = flatten_api_spec(api_spec)
        texts.append(flattened_text)
        labels.append(label)

    print(f"Successfully loaded {len(texts)} API specs.")
    return texts, labels


def flatten_api_spec(api_spec):
    """Flattens an OpenAPI spec into a single string for ML-friendly text representation."""
    desc = f"API: {api_spec.get('info', {}).get('title', '')}\n"

    for path, path_item in api_spec.get('paths', {}).items():
        desc += f"Path: {path}\n"
        for method, method_item in path_item.items():
            if method.lower() in ["get", "post", "put", "delete"]:
                desc += f"Method: {method}\n"

                if "security" in method_item:
                    desc += "Security: authenticated\n"
                else:
                    desc += "Security: none\n"

                if "parameters" in method_item:
                    param_names = [p.get("name", "") for p in method_item["parameters"]]
                    desc += f"Parameters: {', '.join(param_names)}\n"

                desc += f"Description: {method_item.get('description', '')}\n"
    return desc


if __name__ == "__main__":
    texts, labels = load_custom_dataset()
    if texts:
        print(f"\nLoaded {len(texts)} API specifications.")
        print(f"Example flattened spec:\n{texts[0][:500]}...")
        print(f"Example label: {labels[0]}")
