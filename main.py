# main.py
# This is the main execution script for our improved evaluation pipeline.
# It integrates Module 1 (Classifier) and Module 2 (Reliable Rater)
# and compares the output against the original scores in the dataset.

import json
from pathlib import Path
import argparse

# --- 1. Import all necessary functions from our modules ---
from LLMBasedClas import classify_model_response
from ReliableRater import get_reliable_score_from_jury


# --- LLMBasedEvalv1 (baseline) is no longer needed and has been removed ---

def run_improved_evaluation_pipeline(record: dict, config: dict) -> dict:
    """
    Executes our improved evaluation pipeline for a single data record.
    This includes classification and running our improved multi-agent rater.

    Args:
        record: A dictionary representing one row from our dataset.
        config: The global configuration dictionary.

    Returns:
        A dictionary containing the results from our improved evaluation.
    """
    model_response = record['First Model Response']

    # --- Step 1: Classification (Module 1) ---
    print("  - Step 1: Classifying model response...")
    classification = classify_model_response(model_response, config, record['ID'])
    print(f"  - Result: Classified as '{classification}'")

    # --- Step 2: Routing based on classification ---
    if classification == "Clarifying Question":
        print("  - Step 2: Response is a question. Proceeding to evaluation with our improved rater.")

        # --- Run Improved Evaluator (Module 2) ---
        improved_result = get_reliable_score_from_jury(
            modified_problem=record['Modified Description'],
            original_problem=record['Original Description'],
            clarifying_questions=model_response,
            config=config,
            self_consistency_samples=3
        )
        # Add classification to the result for a complete picture
        improved_result['classification'] = classification

    elif classification == "Code Solution":
        print("  - Step 2: Response is a code solution. Bypassing evaluator.")
        note = "Response classified as a code solution; evaluation bypassed to prevent false recovery."
        # The result is a default score of 1, as per our improved mechanism
        improved_result = {
            "classification": classification,
            "final_score": "",
            "agent_votes": {},
            "note": note
        }

    else:  # Handles "Empty Response" or "Classification Failed"
        print(f"  - Step 2: Skipping evaluation due to classification '{classification}'.")
        note = f"Evaluation skipped because response was classified as '{classification}'."
        improved_result = {
            "classification": classification,
            "final_score": "",
            "agent_votes": {},
            "note": note
        }

    return improved_result


# --- Main Execution Block ---
if __name__ == '__main__':

    # --- Main Execution Block ---
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Run improved evaluation on all .jsonl files in EvalInput.")
        parser.add_argument("--input_dir", type=str, default="EvalInput", help="Directory with input .jsonl files")
        parser.add_argument("--output_dir", type=str, default="EvalResults", help="Directory to save results")
        parser.add_argument("--samples", type=int, default=10,
                            help="Number of records to process per file (-1 for all)")
        args = parser.parse_args()

        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Load configuration ---
        try:
            config_path = Path("config.json")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print("--- Configuration loaded successfully. ---")
        except FileNotFoundError as e:
            print(f"FATAL ERROR: {e}.")
            exit()

        # --- 2. Iterate over all .jsonl files ---
        for dataset_path in input_dir.glob("*.jsonl"):
            print(f"\n=== Processing file: {dataset_path.name} ===")

            with open(dataset_path, 'r', encoding='utf-8') as f:
                all_records = [json.loads(line) for line in f]

            if args.samples == -1:
                records_to_process = all_records
            else:
                records_to_process = all_records[:args.samples]

            print(f"Loaded {len(all_records)} records. Evaluating {len(records_to_process)} records...")

            final_results_with_comparison = []
            for i, record in enumerate(records_to_process):
                print(f"--- Processing Record {i + 1}/{len(records_to_process)} (ID: {record['ID']}) ---")
                pipeline_result = run_improved_evaluation_pipeline(record, config)
                record['Improved_Evaluation_Result'] = pipeline_result
                final_results_with_comparison.append(record)
                print("-" * 60)

            # --- 3. Save results ---
            parts = dataset_path.stem.split("_")
            model_name = None
            for i, p in enumerate(parts):
                if p == "model" and i + 1 < len(parts):
                    model_name = parts[i + 1]
                    break
            if model_name is None:
                model_name = dataset_path.stem  # fallback

            output_file = output_dir / f"Eval_Result_{model_name}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in final_results_with_comparison:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            print(f"=== Finished {dataset_path.name}, results saved to {output_file} ===")