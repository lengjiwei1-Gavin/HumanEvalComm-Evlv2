import hashlib
import re
import random
from pathlib import Path
from collections import defaultdict


def extract_unique_id(problem_header: str) -> str:
    """
    Extract unique ID from the problem block header line.
    Example: "****** new problem (name=HumanEval/163 input_prompt=prompt2ap) ******"
    Returns: "HumanEval/163_prompt2ap"
    """
    name_match = re.search(r"name=([\w/]+)", problem_header)
    prompt_match = re.search(r"input_prompt=(\w+)", problem_header)

    if name_match and prompt_match:
        name = name_match.group(1).replace('/', '_')
        prompt_type = prompt_match.group(1)
        return f"{name}_{prompt_type}"
    return None


def extract_model_name(filename: str) -> str:
    """
    Extract model name from filename.
    Example: "manualRemove_dataset_HumanEvalComm_model_Okanagan_topn_1_temperature_1.0.log_1"
    Returns: "Okanagan"
    """
    # Match the part after "model_" and before "_topn"
    match = re.search(r"model_([^_]+)_topn", filename)
    if match:
        return match.group(1)
    return "Unknown"


def mixed_sampling_strategy(problems_with_hash, k: int, random_seed: int = 42):
    """
    Mixed sampling strategy:
    1. First select some by hash value sorting
    2. Then randomly select some
    3. Ensure sample diversity
    """
    random.seed(random_seed)

    # Sort by hash value
    sorted_problems = sorted(problems_with_hash, key=lambda x: x[0])

    # Stratified sampling: 60% hash selection + 40% random selection
    hash_count = int(k * 0.6)
    random_count = k - hash_count

    # Hash selection: evenly distributed across the entire range
    hash_selected = []
    if hash_count > 0:
        step = max(1, len(sorted_problems) // hash_count)
        for i in range(0, len(sorted_problems), step):
            if len(hash_selected) < hash_count:
                hash_selected.append(sorted_problems[i])

    # Random selection: select from remaining problems
    remaining = [p for p in sorted_problems if p not in hash_selected]
    if random_count > 0 and remaining:
        random_selected = random.sample(remaining, min(random_count, len(remaining)))
    else:
        random_selected = []

    # Merge results
    final_selection = hash_selected + random_selected
    return [item[1] for item in final_selection]


def process_single_log(input_path: Path, k: int) -> tuple:
    """
    Process a single log file, return selected samples and some statistics.
    """
    print(f"Processing file: {input_path.name}")

    try:
        content = input_path.read_text(encoding='utf-8', errors='ignore')
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
        return [], {}

    # Use more precise regex to match problem block delimiters
    delimiter_pattern = r"\*{10,}\s+\*{6,}\s+new problem"
    matches = list(re.finditer(delimiter_pattern, content))

    if not matches:
        print(f"  No problem blocks found")
        return [], {}

    problems = []
    # Extract each problem block
    for i, match in enumerate(matches):
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        block = content[start_pos:end_pos].strip()
        problems.append(block)

    print(f"  Found {len(problems)} problem blocks")

    # Calculate hash value for each problem block and check for duplicates
    hashed_problems = []
    seen_ids = set()
    duplicate_count = 0

    for block in problems:
        header_lines = block.split('\n', 2)[:2]
        header_text = "\n".join(header_lines)
        unique_id = extract_unique_id(header_text)

        if unique_id:
            if unique_id in seen_ids:
                duplicate_count += 1
                print(f"  Skipping duplicate ID: {unique_id}")
                continue

            seen_ids.add(unique_id)
            hasher = hashlib.sha256(unique_id.encode('utf-8'))
            hash_int = int(hasher.hexdigest(), 16)
            hashed_problems.append((hash_int, block, unique_id))
        else:
            print(f"  Warning: Unable to extract ID from header line, skipped")

    print(f"  Valid problem blocks after deduplication: {len(hashed_problems)} (duplicates skipped: {duplicate_count})")

    # Check sample count
    if k > len(hashed_problems):
        print(f"  Warning: Requested sample count ({k}) is greater than available problems ({len(hashed_problems)})")
        k = len(hashed_problems)

    # Use mixed sampling strategy
    selected_samples = mixed_sampling_strategy(hashed_problems, k)

    stats = {
        'total_blocks': len(problems),
        'valid_blocks': len(hashed_problems),
        'duplicates_skipped': duplicate_count,
        'selected_count': len(selected_samples)
    }

    return selected_samples, stats


def batch_process_logs(input_dir: Path, sample_count: int = 120):
    """
    Batch process all log files in the directory.
    """
    print(f"=== Starting batch processing of all log files in {input_dir} directory ===")

    # Find all log files (supports .log and .log_* formats)
    log_files = list(input_dir.glob("*.log*"))

    if not log_files:
        print(f"No log files found in directory {input_dir}")
        return

    print(f"Found {len(log_files)} log files")

    # Create output directory
    output_dir = input_dir.parent / "selected_logs"
    output_dir.mkdir(exist_ok=True)

    total_stats = defaultdict(int)
    processed_files = []

    for log_file in log_files:
        # Extract model name
        model_name = extract_model_name(log_file.name)

        # Generate output filename
        output_filename = f"selected_dataset_HumanEvalComm_model_{model_name}_topn_1_temperature_1.0.txt"
        output_path = output_dir / output_filename

        # Process single file
        selected_samples, stats = process_single_log(log_file, sample_count)

        if selected_samples:
            # Write to output file
            output_content = "\n\n".join(selected_samples)
            output_path.write_text(output_content, encoding='utf-8')

            print(f"  ✓ Successfully processed, output to: {output_filename}")
            print(f"    Stats: total_blocks={stats['total_blocks']}, valid_blocks={stats['valid_blocks']}, "
                  f"duplicates_skipped={stats['duplicates_skipped']}, selected={stats['selected_count']}")

            # Accumulate statistics
            for key, value in stats.items():
                total_stats[key] += value

            processed_files.append({
                'input_file': log_file.name,
                'output_file': output_filename,
                'model_name': model_name,
                **stats
            })
        else:
            print(f"  ✗ Processing failed: {log_file.name}")

    # Print overall statistics
    print(f"\n=== Batch processing completed ===")
    print(f"Successfully processed files: {len(processed_files)}")
    print(f"Output directory: {output_dir}")
    print(f"Overall statistics:")
    print(f"  - Total problem blocks: {total_stats['total_blocks']}")
    print(f"  - Total valid blocks: {total_stats['valid_blocks']}")
    print(f"  - Total duplicates skipped: {total_stats['duplicates_skipped']}")
    print(f"  - Total selected samples: {total_stats['selected_count']}")

    # Generate processing report
    report_path = output_dir / "processing_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Batch Processing Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Processing time: {Path().cwd()}\n")
        f.write(f"Input directory: {input_dir}\n")
        f.write(f"Sample count: {sample_count}\n\n")

        f.write("Processing results:\n")
        for file_info in processed_files:
            f.write(f"\nFile: {file_info['input_file']}\n")
            f.write(f"  Model: {file_info['model_name']}\n")
            f.write(f"  Output: {file_info['output_file']}\n")
            f.write(f"  Stats: total_blocks={file_info['total_blocks']}, "
                    f"valid_blocks={file_info['valid_blocks']}, "
                    f"duplicates={file_info['duplicates_skipped']}, "
                    f"selected={file_info['selected_count']}\n")

        f.write(f"\nOverall statistics:\n")
        f.write(f"  Successfully processed files: {len(processed_files)}\n")
        for key, value in total_stats.items():
            f.write(f"  {key}: {value}\n")

    print(f"Processing report saved to: {report_path}")


if __name__ == "__main__":

    INPUT_DIR = Path("OriginalLog-from-HumanEvalComm")
    SAMPLE_COUNT = 120

    if not INPUT_DIR.exists():
        print(f"Error: Input directory '{INPUT_DIR}' does not exist.")
        print("Please ensure the directory path is correct.")
    elif not INPUT_DIR.is_dir():
        print(f"Error: '{INPUT_DIR}' is not a directory.")
    else:
        batch_process_logs(INPUT_DIR, SAMPLE_COUNT)

