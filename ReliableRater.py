# ReliableRater.py
# Module 2: This script implements a robust, multi-agent, self-consistent
# scoring mechanism for evaluating the quality of clarifying questions.

import time
import json
import requests
from collections import Counter
from pathlib import Path
import statistics

# --- 1. Import prompt templates from the central prompts file ---
from prompts import IMPROVED_EVALUATOR_PROMPT_TEMPLATE


# --- 2. Core Functions ---

def _call_single_evaluator(prompt: str, agent_config: dict, config: dict) -> int | None:
    """
    Makes a single API call to a specified LLM evaluator and parses the score.
    This function is now a generic dispatcher for different API providers.
    """
    provider = agent_config.get('provider')
    model = agent_config.get('model')
    name = agent_config.get('name')

    if not provider:
        print(f"    - [Rater ERROR]: Provider not specified for agent '{name}'.")
        return None

    api_key = config['API_KEYS'].get(provider)
    api_url = config['API_URLS'].get(provider)

    if not api_key or api_key.startswith("YOUR_"):
        print(f"    - [Rater ERROR]: API key for provider '{provider}' is not configured in config.json.")
        return None
    if not api_url:
        print(f"    - [Rater ERROR]: API URL for provider '{provider}' is not configured in config.json.")
        return None

    headers = {'Content-Type': 'application/json'}

    # --- Build requests tailored for different service providers ---
    if provider in ['openai', 'deepseek']:
        headers['Authorization'] = f'Bearer {api_key}'
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0,
            "response_format": {"type": "json_object"}  # JSON output
        }

    elif provider == 'claude':
        headers['x-api-key'] = api_key
        headers['anthropic-version'] = '2023-06-01'
        headers['Content-Type'] = 'application/json'

        payload = {
            "model": model,
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0,
            "tools": [{
                "name": "provide_score",
                "description": "Provide evaluation score",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "integer",
                            "description": "Evaluation score (1, 2, or 3)",
                            "enum": [1, 2, 3]
                        }
                    },
                    "required": ["score"]
                }
            }],
            "tool_choice": {"type": "tool", "name": "provide_score"}
        }


    else:
        print(f"    - [Rater ERROR]: Provider '{provider}' is not supported by the current script logic.")
        return None

    try:
        # --- Call API ---
        response = requests.post(api_url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()  # 如果请求失败 (如 4xx or 5xx), 将抛出异常
        api_response = response.json()

        # --- Analyzing responses from different service providers ---
        score = None
        response_content_str = ""
        if provider in ['openai', 'deepseek']:
            # 路径: response['choices'][0]['message']['content']
            response_content_str = api_response.get('choices', [{}])[0].get('message', {}).get('content', '{}')

        elif provider == 'claude':
            if 'content' in api_response and api_response['content']:
                content = api_response['content'][0]
                if content.get('type') == 'tool_use':
                    tool_input = content.get('input', {})
                    response_content_str = json.dumps(tool_input)
                else:
                    response_content_str = content.get('text', '{}')
            else:
                response_content_str = '{}'

        # Parse the response string into a JSON object and extract the score.
        if response_content_str:
            response_content_json = json.loads(response_content_str)
            score = response_content_json.get("score")
            print(f"      - [API Call]: {name} returned score: {score}")
        else:
            print(f"    - [Rater WARNING]: Empty response content from {name}.")

        if isinstance(score, int) and score in [1, 2, 3]:
            return score
        else:
            print(f"    - [Rater WARNING]: Invalid or missing score '{score}' received from {name}.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"    - [Rater ERROR]: API network call for {name} failed: {e}")
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"    - [Rater ERROR]: Failed to parse response from {name}. Error: {e}. Raw response: {response.text}")
        return None
    except Exception as e:
        print(f"    - [Rater ERROR]: An unexpected error occurred for {name}: {e}")
        return None

def get_self_consistent_vote(agent_config: dict, config: dict, modified_problem: str, original_problem: str,
                             clarifying_questions: str, n_samples: int) -> int | None:
    """
    Gets a single agent's reliable vote through self-consistency.
    """
    scores = []
    prompt = IMPROVED_EVALUATOR_PROMPT_TEMPLATE.format(
        original_problem=original_problem,
        modified_problem=modified_problem,
        clarifying_questions=clarifying_questions
    )

    print(f"  - [Self-Consistency]: Polling {agent_config['name']} {n_samples} times...")
    for i in range(n_samples):
        score = _call_single_evaluator(prompt, agent_config, config)
        if score is not None:
            scores.append(score)
        time.sleep(1.5)  # A small delay to respect potential rate limits

    if not scores:
        print(f"    - [Self-Consistency]: All polls for {agent_config['name']} failed.")
        return None

    # Aggregate scores for this agent using the "majority, median fallback" rule
    counts = Counter(scores)
    most_common = counts.most_common(1)

    if most_common and most_common[0][1] > len(scores) / 2:
        vote = most_common[0][0]
        print(f"    - [Self-Consistency]: Majority vote for {agent_config['name']} on {scores} -> {vote}")
    else:
        vote = int(round(statistics.median(scores)))
        print(
            f"    - [Self-Consistency]: No majority for {agent_config['name']} in {scores}. Median fallback -> {vote}")

    return vote


# --- 3. Main Function for this Module ---

def get_reliable_score_from_jury(modified_problem: str, original_problem: str, clarifying_questions: str, config: dict,
                                 self_consistency_samples: int = 3) -> dict:
    """
    Gets a final, reliable score by polling a jury of LLM agents, each of which
    uses self-consistency to determine its vote.

    Returns:
        A dictionary with the final score and details of the voting process.
    """
    agent_votes = {}
    jury = config['MODELS']['jury']

    # Step 1: Each agent in the jury determines its self-consistent vote
    for agent in jury:
        vote = get_self_consistent_vote(
            agent_config=agent,
            config=config,
            modified_problem=modified_problem,
            original_problem=original_problem,
            clarifying_questions=clarifying_questions,
            n_samples=self_consistency_samples
        )
        agent_votes[agent['name']] = vote

    # Step 2: Aggregate the final votes from all agents
    final_votes = [v for v in agent_votes.values() if v is not None]

    if not final_votes:
        final_score = 1  # Default to lowest score if all agents fail
        note = "All agents failed to provide a vote."
    else:
        counts = Counter(final_votes)
        most_common = counts.most_common(1)
        if most_common and most_common[0][1] > len(final_votes) / 2:
            final_score = most_common[0][0]
            note = f"Final score decided by majority vote on {final_votes}."
        else:
            final_score = int(round(statistics.median(final_votes)))
            note = f"No majority in {final_votes}, used median fallback."

    return {
        "final_score": final_score,
        "agent_votes": agent_votes,
        "note": note
    }


# --- 4. Standalone Demonstration Block (MODIFIED FOR BATCH PROCESSING) ---

if __name__ == '__main__':
    # --- 1. Load configuration ---
    try:
        config_path = Path("config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print("--- Configuration loaded successfully. ---")
    except FileNotFoundError as e:
        print(f"FATAL ERROR: A required file was not found: {e}.")
        exit()

    # --- 2. Setup for batch processing ---
    NUM_SAMPLES_TO_TEST = 10

    dataset_path = Path("EvalInput/selected_dataset_HumanEvalComm_model_CodeLlama-13b-Instruct-hf_topn_1_temperature_1.0.jsonl")
    all_results = []

    print(f"\n--- Running Demonstration on the first {NUM_SAMPLES_TO_TEST} samples from {dataset_path.name} ---")

    # --- 3. Process each sample record in a loop ---
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= NUM_SAMPLES_TO_TEST:
                    break

                sample_record = json.loads(line)
                record_id = sample_record.get('ID', f'Record_{i + 1}')

                print(f"\n--- [Processing Sample {i + 1}/{NUM_SAMPLES_TO_TEST}] ID: {record_id} ---")

                result = get_reliable_score_from_jury(
                    modified_problem=sample_record['Modified Description'],
                    original_problem=sample_record['Original Description'],
                    clarifying_questions=sample_record['First Model Response'],
                    config=config,
                    self_consistency_samples=3
                )

                all_results.append({
                    "id": record_id,
                    "final_score": result["final_score"],
                    "agent_votes": result["agent_votes"],
                    "note": result["note"]
                })

                print(f"--- [Finished Sample {i + 1}] Final Score: {result['final_score']} ---")

    except FileNotFoundError:
        print(f"FATAL ERROR: Dataset file not found at {dataset_path}.")
        exit()
    except Exception as e:
        print(f"An error occurred during processing: {e}")

    # --- 4. Print final summary ---
    print("\n\n--- FINAL BATCH SUMMARY ---")
    if not all_results:
        print("No samples were processed.")
    else:
        final_scores = []
        for res in all_results:
            print(f"  - Sample ID: {res['id']}, Final Score: {res['final_score']}, Votes: {res['agent_votes']}")
            final_scores.append(res['final_score'])
