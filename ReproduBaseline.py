# ReproduBaseline.py
# This script provides a faithful, modular implementation of the original
# LLM-based evaluator from the HumanEvalComm paper for baseline comparison.

import json
import re
import requests
import time
from pathlib import Path

# --- 1. Import necessary components from other modules ---
from prompts import ORIGINAL_EVALUATOR_PROMPT_TEMPLATE


# --- 2. Core Functions ---

def evaluate_clarification_request(
        modified_problem: str,
        original_problem: str,
        clarifying_questions: str,
        config: dict
) -> dict:
    """
    Evaluates a model's clarification question by calling the OpenAI API,
    using the original paper's methodology.
    """
    api_key = config['API_KEYS'].get('openai')
    api_url = config['API_URLS'].get('openai')
    # Per the original paper's implementation details
    model_name = "gpt-3.5-turbo"

    if not api_key or api_key.startswith("YOUR_"):
        return {"error": "OpenAI API key is not configured in config.json."}

    prompt = ORIGINAL_EVALUATOR_PROMPT_TEMPLATE.format(
        clarifying_questions=clarifying_questions,
        modified_problem=modified_problem,
        original_problem=original_problem,
        answer="{answer}"  # This is a placeholder for the template
    )

    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
    }

    print("    - [Baseline Eval]: Calling OpenAI Evaluator LLM...")
    try:
        # --- SIMULATION FOR LOCAL TESTING ---
        # To run without live API keys, this part is simulated.
        # In a real run, you would comment out the simulation block
        # and uncomment the actual API call logic.


        # --- REAL API CALL LOGIC (Commented out for now) ---
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        api_response = response.json()
        response_text = api_response['choices'][0]['message']['content']

        quality_match = re.search(r"QUALITY=(\d+)", response_text)
        answer_match = re.search(r'ANSWERS=([\'"`]{1,3})(.*?)\1', response_text, re.DOTALL)
        quality = int(quality_match.group(1)) if quality_match else None
        answer = answer_match.group(2).strip() if answer_match else "Parsing Failed"
        return {"quality": quality, "answer": answer}

    except Exception as e:
        return {"error": f"API call failed: {e}"}


def response_2_code_if_no_text(response: str) -> str:
    """
    The original project's mechanism. It only extracts code if the ENTIRE
    response string is a single markdown code block.
    """
    code_template = re.compile(r'^\s*```.*?\n([\s\S]+?)\n```\s*$', re.M)
    match = code_template.match(response)
    if match:
        return match.group(1)
    return ""


def is_empty_response(response: str) -> bool:
    """
    Check if the response is empty or contains only whitespace.
    Added to handle empty responses properly.
    """
    return not response or response.strip() == ""


# --- 3. Main Execution Block (MODIFIED TO PERFORM FULL EVALUATION) ---

if __name__ == '__main__':

    # --- Step 1: Load Configuration and Data ---
    try:
        config_path = Path("config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        dataset_path = Path("EvalInput/selected_dataset_HumanEvalComm_model_CodeLlama-13b-Instruct-hf_topn_1_temperature_1.0.jsonl")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            all_records = [json.loads(line) for line in f]

        print(f"--- Successfully loaded config and {len(all_records)} records. ---")
    except FileNotFoundError as e:
        print(f"FATAL ERROR: A required file was not found: {e}.")
        exit()

    records_to_process = all_records[:5]
    print(f"--- Preparing to evaluate the first {len(records_to_process)} records using the baseline method. ---\n")

    # --- Step 2: Main Evaluation Loop ---
    baseline_results = []
    for i, record in enumerate(records_to_process):
        print(f"--- Processing Record {i + 1}/{len(records_to_process)} (ID: {record['ID']}) ---")

        model_response = record['First Model Response']
        final_result = {}

        # --- Step 2a: Check for empty response first ---
        if is_empty_response(model_response):
            # If the response is empty, assign empty scores
            print("  - Empty Response Check: Response is completely empty.")
            print("  - Action: Assigning empty scores.")
            final_result = {
                "quality": "",
                "answer": "",
                "note": "Empty response - no evaluation performed."
            }
        else:
            # --- Step 2b: Apply the original project's trigger mechanism ---
            extracted_code = response_2_code_if_no_text(model_response)

            if extracted_code:
                # If the mechanism extracts code, the evaluator is NOT triggered.
                print("  - Mechanism Check: Response is a pure code block.")
                print("  - Action: Bypassing evaluator.")
                final_result = {
                    "quality": "",  # In the original paper, this would be a default '' score
                    "answer": "",
                    "note": "Evaluator not triggered as per original mechanism."
                }
            else:
                # If the mechanism fails, the evaluator IS triggered.
                print("  - Mechanism Check: Response is NOT a pure code block.")
                print("  - Action: Triggering evaluator.")
                final_result = evaluate_clarification_request(
                    modified_problem=record['Modified Description'],
                    original_problem=record['Original Description'],
                    clarifying_questions=model_response,
                    config=config
                )

        record['Baseline_Evaluation_Result'] = final_result
        baseline_results.append(record)
        time.sleep(0.5)  # Simulate delay

    # --- Step 3: Print Final Report ---
    print("\n\n--- === BASELINE EVALUATION REPORT === ---")
    for record in baseline_results:
        original_score_from_log = record.get('Evaluator Quality Score', 'N/A')
        new_run_result = record['Baseline_Evaluation_Result']

        print(f"\n--- ID: {record['ID']} ---")
        print(f"  - Original Paper's Score (from log): {original_score_from_log}")
        print(f"  - Our Baseline Run Score           : {new_run_result.get('quality', 'N/A')}")
        print(f"  - Evaluator Answer                 : {new_run_result.get('answer', 'N/A')}")
        if 'note' in new_run_result:
            print(f"  - Note                           : {new_run_result['note']}")