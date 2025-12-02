import json
import difflib

# Load JSON once
with open("animal_info.json", "r") as f:
    ANIMAL_INFO = json.load(f)


def normalize(name: str):
    """Normalize keys for matching."""
    return name.lower().strip().replace("-", " ").replace("_", " ")


def get_animal_info(name: str):
    """
    Return dictionary containing animal information.
    Ensures no missing or undefined fields.
    """

    # Default values always present
    DEFAULT = {
        "scientific_name": "Information not available",
        "habitat": "Information not available",
        "life_span": "Information not available",   # FIXED naming
        "lifespan": "Information not available",    # support both keys
        "diet": "Information not available",
        "other_info": "Information not available"
    }

    # If prediction empty → return default safely
    if not name:
        return DEFAULT

    name = normalize(name)

    # 1. Exact match
    if name in ANIMAL_INFO:
        return merge_with_defaults(DEFAULT, ANIMAL_INFO[name])

    # 2. Partial match (substring)
    for key in ANIMAL_INFO.keys():
        if name in key or key in name:
            return merge_with_defaults(DEFAULT, ANIMAL_INFO[key])

    # 3. Fuzzy match (spelling mistakes)
    best = difflib.get_close_matches(name, ANIMAL_INFO.keys(), n=1, cutoff=0.55)
    if best:
        return merge_with_defaults(DEFAULT, ANIMAL_INFO[best[0]])

    # If nothing matches, return defaults
    return DEFAULT


def merge_with_defaults(defaults, data):
    """
    Ensures ALL expected fields exist.
    Also auto-fixes JSON keys like "lifespan" vs "life_span".
    """
    clean = defaults.copy()
    clean.update(data)

    # If JSON has "lifespan", move to "life_span"
    if "lifespan" in data and data["lifespan"]:
        clean["life_span"] = data["lifespan"]

    return clean
