import random
import requests
import json
from pathlib import Path

# --- 1. Modularization: Import the prompt from an external file ---
from prompts import CLASSIFIER_PROMPT_TEMPLATE


# --- 2. Core Functions (Adapted for Modularity and Internationalization) ---

def is_empty_or_invalid_response(response_text: str) -> bool:
    """
    Checks if the response is empty or invalid.
    (This function's logic is preserved from your version, with Chinese patterns removed).
    """
    if not response_text:
        return True
    cleaned_text = response_text.strip()
    if not cleaned_text:
        return True
    meaningless_patterns = [
        "n/a", "na", "none", "null", "empty",
        "no response", "no content", "-", "--", "---"
    ]
    if cleaned_text.lower() in meaningless_patterns:
        return True
    if all(c in " \t\n\r.,;:!?-_()[]{}\"'" for c in cleaned_text):
        return True
    return False


def classify_model_response(response_text: str, config: dict, record_id: str = "Unknown") -> str:
    """
    Uses an LLM to classify a model's response.
    (This function is now modularized to get API settings from the config dictionary).
    """
    if is_empty_or_invalid_response(response_text):
        print(f"Record {record_id}: Detected empty or invalid response.")
        return "Empty Response"

    # --- Modularization: Load configuration from the config dictionary ---
    api_key = config['API_KEYS'].get('openai')
    api_url = config['API_URLS'].get('openai')
    model_name = config['MODELS'].get('classifier')

    if not api_key or api_key.startswith("YOUR_"):
        print("ERROR: Classifier API key is not configured in config.json.")
        return "Classification Failed"

    prompt = CLASSIFIER_PROMPT_TEMPLATE.format(code_llm_1st_response=response_text)
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 50
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        api_response = response.json()
        classification_raw = api_response['choices'][0]['message']['content'].strip()

        try:
            classification_json = json.loads(classification_raw)
            classification = classification_json.get('classification', '')
        except json.JSONDecodeError:
            classification = classification_raw

        if classification in ["Clarifying Question", "Code Solution"]:
            return classification
        else:
            print(f"Warning: Abnormal classification result for record {record_id}: {classification_raw}")
            return "Code Solution" if "```" in response_text else "Clarifying Question"

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Classifier API call failed for record {record_id}: {e}")
        return "Classification Failed"


def analyze_dataset_quality(all_records):
    """
    Analyzes dataset quality by counting empty/invalid responses.
    (This function's logic is preserved from your version, with print statements translated).
    """
    total_records = len(all_records)
    empty_responses = 0
    valid_responses = 0
    print(f"\n=== Dataset Quality Analysis ===")
    print(f"Total Records: {total_records}")
    for record in all_records:
        response_text = record.get('First Model Response', '')
        if is_empty_or_invalid_response(response_text):
            empty_responses += 1
        else:
            valid_responses += 1
    print(f"Empty/Invalid Responses: {empty_responses} ({empty_responses / total_records * 100:.1f}%)")
    print(f"Valid Responses: {valid_responses} ({valid_responses / total_records * 100:.1f}%)")
    print("=" * 30)


# --- 3. Main Execution Block (Logic preserved from your version) ---

if __name__ == '__main__':

    # --- Modularization: First, load the configuration file ---
    try:
        config_path = Path("config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("--- Configuration successfully loaded from config.json ---")
    except FileNotFoundError:
        print("FATAL ERROR: config.json not found. Please create it according to the project plan.")
        exit()

    # --- Step 1: Define data source and load all records (logic preserved) ---
    dataset_file = Path("EvalInput/selected_dataset_HumanEvalComm_model_CodeLlama-13b-Instruct-hf_topn_1_temperature_1.0.jsonl")
    if not dataset_file.exists():
        print(f"Error: Dataset file not found: '{dataset_file}'")
    else:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            all_records = [json.loads(line) for line in f]
        print(f"--- Successfully loaded {len(all_records)} records from {dataset_file} ---")

    analyze_dataset_quality(all_records)
    valid_records = [r for r in all_records if not is_empty_or_invalid_response(r.get('First Model Response', ''))]

    if not valid_records:
        print("Warning: No valid response data found for testing.")
        exit()

    sample_size = min(10, len(valid_records))
    random_records = random.sample(valid_records, sample_size)
    print(f"\n--- Randomly testing {len(random_records)} valid samples ---\n")

    classification_stats = {
        "Clarifying Question": 0, "Code Solution": 0,
        "Empty Response": 0, "Classification Failed": 0
    }

    for i, record in enumerate(random_records):
        record_id = record.get('ID', f'Record_{i + 1}')
        print(f"--- Test Case {i + 1}: {record_id} ---")
        response_text = record.get('First Model Response', '')

        # --- Modularization: Pass the config dictionary to the function ---
        classification = classify_model_response(response_text, config, record_id)

        classification_stats[classification] = classification_stats.get(classification, 0) + 1
        print(f"  - Classification: {classification}")

        if classification == "Empty Response":
            print(f"  - Response Content: [Empty Response]")
        else:
            preview_length = 100 if classification == "Code Solution" else 150
            print(f"  - Response Preview: {response_text[:preview_length]}...")
        print()

    print("\n=== Classification Stats Summary ===")
    for category, count in classification_stats.items():
        percentage = count / len(random_records) * 100
        print(f"{category}: {count} ({percentage:.1f}%)")