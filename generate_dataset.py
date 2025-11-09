import os
import json
import random
import yaml

DATASET_DIR = os.path.join(os.getcwd(), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

MANIFEST_FILE = "custom_dataset_150_balanced.json"
TOTAL_FILES = 150

# Tuned realism: less label noise and overlap
NOISE_RATE = 0.02       # 2% mislabeling
OVERLAP_RATE = 0.05     # 5% overlapping safe/vulnerable features

SAFE_KEYWORDS = [
    "validate", "sanitize", "encrypt", "token", "jwt", "secure", "csrf_protected",
    "rate_limited", "auth_required", "parameterized_query", "input_filter"
]

VULN_KEYWORDS = [
    "bypass", "injection", "xss", "exposed", "leak", "plaintext", "hardcoded", "unsafe_query",
    "unvalidated", "debug_enabled", "missing_auth"
]

SAFE_TEMPLATES = [
    {
        "api_name": "Secure Auth API",
        "description": "Provides safe authentication and token-based access control.",
        "paths": {
            "/login": {"post": {"summary": "Validates credentials securely"}},
            "/token": {"get": {"summary": "Returns encrypted JWT access token"}}
        }
    },
    {
        "api_name": "Protected Payment Gateway",
        "description": "Handles payment with encryption and secure endpoints.",
        "paths": {
            "/pay": {"post": {"summary": "Processes transactions via SSL"}},
            "/refund": {"post": {"summary": "Verifies refund requests with digital signatures"}}
        }
    },
    {
        "api_name": "Validated User Service",
        "description": "Ensures all user data is validated before storage.",
        "paths": {
            "/user": {"post": {"summary": "Validates and stores user data"}},
            "/user/{id}": {"get": {"summary": "Retrieves user securely"}}
        }
    }
]

VULN_TEMPLATES = [
    {
        "api_name": "Insecure Admin Panel",
        "description": "Admin interface without authentication or input validation.",
        "paths": {
            "/admin": {"get": {"summary": "Accessible admin dashboard without login"}},
            "/config": {"post": {"summary": "Allows configuration changes by any user"}}
        }
    },
    {
        "api_name": "SQL Injection API",
        "description": "Accepts unescaped user input directly in SQL queries.",
        "paths": {
            "/search": {"get": {"summary": "Concatenates query strings from parameters"}}
        }
    },
    {
        "api_name": "XSS-prone Comment API",
        "description": "Displays unfiltered HTML comments in responses.",
        "paths": {
            "/comment": {"post": {"summary": "Echoes user input without sanitization"}}
        }
    }
]

def random_mix(template_a, template_b):
    """Blend safe and vulnerable APIs to create ambiguous cases."""
    mixed = dict(template_a)
    mixed["paths"].update(template_b["paths"])
    mixed["description"] += " Mixed endpoints detected: potential misconfiguration."
    return mixed

def generate_yaml_file(entry_name, content):
    file_path = os.path.join(DATASET_DIR, f"{entry_name}.yaml")
    with open(file_path, "w") as f:
        yaml.dump(content, f, sort_keys=False)
    return file_path

def main():
    print("Generating balanced dataset (low noise, realistic distinction)...")

    entries = []
    for i in range(TOTAL_FILES):
        is_vuln = random.random() < 0.5
        if random.random() < OVERLAP_RATE:
            safe = random.choice(SAFE_TEMPLATES)
            vuln = random.choice(VULN_TEMPLATES)
            template = random_mix(safe, vuln)
        else:
            template = random.choice(VULN_TEMPLATES if is_vuln else SAFE_TEMPLATES)

        keywords = random.sample(VULN_KEYWORDS if is_vuln else SAFE_KEYWORDS, k=4)
        template["keywords"] = keywords
        template["description"] += " Keywords: " + ", ".join(keywords)

        entry_name = f"{'vuln' if is_vuln else 'safe'}_{template['api_name'].replace(' ', '_').lower()}_{i+1}"
        generate_yaml_file(entry_name, template)

        # Apply controlled label noise
        if random.random() < NOISE_RATE:
            is_vuln = not is_vuln

        entries.append({
            "api_name": template["api_name"],
            "filename": f"{entry_name}.yaml",
            "vulnerable": is_vuln
        })

    manifest = {"api_specs": entries}
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created {TOTAL_FILES} YAML specs with controlled noise and overlap.")
    print(f"Manifest saved to {MANIFEST_FILE}")

if __name__ == "__main__":
    main()
