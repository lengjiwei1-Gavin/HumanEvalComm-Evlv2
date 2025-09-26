import re
import json
from pathlib import Path
from collections import defaultdict


def extract_field(block: str, start_tag: str, end_tag: str) -> str:
    """
    A safe helper function to extract text between two tags.
    Returns an empty string if tags are not found.
    """
    try:
        content = block.split(start_tag)[1].split(end_tag)[0]
        return content.strip()
    except IndexError:
        return ""


def extract_model_name_from_filename(filename: str) -> str:
    """
    Extract model name from filename.
    Example: "selected_dataset_HumanEvalComm_model_Okanagan_topn_1_temperature_1.0.txt"
    Returns: "Okanagan"
    """
    match = re.search(r"model_([^_]+)_topn", filename)
    if match:
        return match.group(1)
    return "Unknown"


def extract_rq3_data_tolerant(log_filepath: Path) -> tuple:
    """
    Robustly extract structured data for RQ3 reproduction from the specified log file.
    This version handles incomplete records tolerantly, filling empty values for missing fields.

    Args:
        log_filepath: Path to the log file.

    Returns:
        A tuple: (list of extracted data, statistics dictionary)
    """
    print(f"Processing file: {log_filepath.name}")

    try:
        content = log_filepath.read_text(encoding='utf-8', errors='ignore')
    except FileNotFoundError:
        print(f"  Error: Input file not found: {log_filepath}")
        return [], {}

    # Step 1: Use more precise regex from reference script to locate the start of each problem block
    delimiter_pattern = r"\*{10,}\s+\*{6,}\s+new problem"
    matches = list(re.finditer(delimiter_pattern, content))

    if not matches:
        print(f"  No problem blocks found")
        return [], {}

    problem_blocks = []
    for i, match in enumerate(matches):
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        block = content[start_pos:end_pos].strip()
        problem_blocks.append(block)

    print(f"  Found {len(problem_blocks)} problem blocks")

    structured_data = []
    stats = {
        'total_blocks': len(problem_blocks),
        'successful_extractions': 0,
        'partial_extractions': 0,
        'failed_extractions': 0,
        'missing_fields': defaultdict(int)
    }

    for i, block in enumerate(problem_blocks):
        # Step 2: Initialize a dictionary with all fields, default values as empty
        data_point = {
            'ID': "",
            'Modification Type': "",
            'First Model Response': "",
            'Modified Description': "",
            'Original Description': "",
            'Evaluator Quality Score': "",
            'Evaluator Answer': ""
        }

        # Track the number of fields extracted
        fields_extracted = 0
        total_fields = len(data_point)

        # Step 3: Try to extract each field, keep empty if not found
        header_match = re.search(r"new problem \(name=(.*?) input_prompt=(.*?)\)", block)
        if header_match:
            data_point['ID'] = header_match.group(1).strip()
            data_point['Modification Type'] = header_match.group(2).strip()
            if data_point['ID']:
                fields_extracted += 1
            else:
                stats['missing_fields']['ID'] += 1
            if data_point['Modification Type']:
                fields_extracted += 1
            else:
                stats['missing_fields']['Modification Type'] += 1
        else:
            stats['missing_fields']['ID'] += 1
            stats['missing_fields']['Modification Type'] += 1

        # Extract various fields
        first_response = extract_field(block, '!!!!!!!!!!!!! 1st CodeLLM response:',
                                       '!!!!!!!!!!!!! 1st CodeLLM response code:')
        if first_response:
            data_point['First Model Response'] = first_response
            fields_extracted += 1
        else:
            stats['missing_fields']['First Model Response'] += 1

        modified_desc = extract_field(block, '### Modified Problem Description:',
                                      '### Original Description:')
        if modified_desc:
            data_point['Modified Description'] = modified_desc
            fields_extracted += 1
        else:
            stats['missing_fields']['Modified Description'] += 1

        original_desc = extract_field(block, '### Original Description:', '!!!!!!!Completion=')
        if original_desc:
            data_point['Original Description'] = original_desc
            fields_extracted += 1
        else:
            stats['missing_fields']['Original Description'] += 1

        # Extract single-line tags
        try:
            quality_score = block.split('!!!!!!!question_quality_str')[1].strip().split('\n')[0].strip()
            if quality_score:
                data_point['Evaluator Quality Score'] = quality_score
                fields_extracted += 1
            else:
                stats['missing_fields']['Evaluator Quality Score'] += 1
        except IndexError:
            stats['missing_fields']['Evaluator Quality Score'] += 1

        try:
            answer_part = block.split('!!!!!!!answer_str')[1].strip()
            answer = answer_part.split('!!!!!!!')[0].strip()
            if answer:
                data_point['Evaluator Answer'] = answer
                fields_extracted += 1
            else:
                stats['missing_fields']['Evaluator Answer'] += 1
        except IndexError:
            stats['missing_fields']['Evaluator Answer'] += 1

        # Statistics on extraction quality
        if fields_extracted == total_fields:
            stats['successful_extractions'] += 1
        elif fields_extracted > 0:
            stats['partial_extractions'] += 1
        else:
            stats['failed_extractions'] += 1

        structured_data.append(data_point)

    print(f"  Extraction stats: complete={stats['successful_extractions']}, "
          f"partial={stats['partial_extractions']}, "
          f"failed={stats['failed_extractions']}")

    return structured_data, stats


def batch_extract_data(input_dir: Path):
    """
    Batch process all txt files in the directory, extract structured data and save as jsonl format.
    """
    print(f"=== Starting batch extraction of all txt files in {input_dir} directory ===")

    # Find all txt files
    txt_files = list(input_dir.glob("*.txt"))

    if not txt_files:
        print(f"No txt files found in directory {input_dir}")
        return

    print(f"Found {len(txt_files)} txt files")

    # Create output directory
    output_dir = input_dir.parent / "extracted_data"
    output_dir.mkdir(exist_ok=True)

    total_stats = defaultdict(int)
    total_missing_fields = defaultdict(int)
    processed_files = []

    for txt_file in txt_files:
        # Extract model name
        model_name = extract_model_name_from_filename(txt_file.name)

        # Generate output filename (replace .txt with .jsonl)
        output_filename = txt_file.stem + ".jsonl"
        output_path = output_dir / output_filename

        # Extract data
        extracted_data, stats = extract_rq3_data_tolerant(txt_file)

        if extracted_data:
            # Write to JSONL file
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in extracted_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            print(f"  ✓ Successfully extracted and saved to: {output_filename}")
            print(f"    Data count: {len(extracted_data)}")

            # Accumulate statistics
            for key, value in stats.items():
                if key == 'missing_fields':
                    for field, count in value.items():
                        total_missing_fields[field] += count
                else:
                    total_stats[key] += value

            processed_files.append({
                'input_file': txt_file.name,
                'output_file': output_filename,
                'model_name': model_name,
                'data_count': len(extracted_data),
                **{k: v for k, v in stats.items() if k != 'missing_fields'}
            })
        else:
            print(f"  ✗ Extraction failed: {txt_file.name}")

    # Print overall statistics
    print(f"\n=== Batch extraction completed ===")
    print(f"Successfully processed files: {len(processed_files)}")
    print(f"Output directory: {output_dir}")
    print(f"Overall statistics:")
    print(f"  - Total problem blocks: {total_stats['total_blocks']}")
    print(f"  - Complete extractions: {total_stats['successful_extractions']}")
    print(f"  - Partial extractions: {total_stats['partial_extractions']}")
    print(f"  - Failed extractions: {total_stats['failed_extractions']}")

    if total_missing_fields:
        print(f"  - Missing field statistics:")
        for field, count in sorted(total_missing_fields.items()):
            print(f"    {field}: {count}")

    # Generate extraction report
    report_path = output_dir / "extraction_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Batch Data Extraction Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input directory: {input_dir}\n")
        f.write(f"Output directory: {output_dir}\n\n")

        f.write("Processing results:\n")
        for file_info in processed_files:
            f.write(f"\nFile: {file_info['input_file']}\n")
            f.write(f"  Model: {file_info['model_name']}\n")
            f.write(f"  Output: {file_info['output_file']}\n")
            f.write(f"  Data count: {file_info['data_count']}\n")
            f.write(f"  Complete extractions: {file_info['successful_extractions']}\n")
            f.write(f"  Partial extractions: {file_info['partial_extractions']}\n")
            f.write(f"  Failed extractions: {file_info['failed_extractions']}\n")

        f.write(f"\nOverall statistics:\n")
        f.write(f"  Successfully processed files: {len(processed_files)}\n")
        for key, value in total_stats.items():
            f.write(f"  {key}: {value}\n")

        if total_missing_fields:
            f.write(f"\nMissing field statistics:\n")
            for field, count in sorted(total_missing_fields.items()):
                f.write(f"  {field}: {count}\n")

    print(f"Extraction report saved to: {report_path}")

    # Generate combined JSONL file (optional)
    if processed_files:
        combined_output_path = output_dir / "combined_all_models.jsonl"
        total_records = 0

        with open(combined_output_path, 'w', encoding='utf-8') as combined_f:
            for file_info in processed_files:
                model_name = file_info['model_name']
                jsonl_path = output_dir / file_info['output_file']

                # Read each jsonl file and add model name field
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        data['Model_Name'] = model_name  # Add model name field
                        combined_f.write(json.dumps(data, ensure_ascii=False) + '\n')
                        total_records += 1

        print(f"Combined file saved: {combined_output_path} (contains {total_records} records)")


# --- Main execution program ---
if __name__ == '__main__':
    # Input directory path
    INPUT_DIR = Path("selected_logs")

    # Check if input directory exists
    if not INPUT_DIR.exists():
        print(f"Error: Input directory '{INPUT_DIR}' does not exist.")
        print("Please ensure LogSelector has been run to generate selected log files.")
    elif not INPUT_DIR.is_dir():
        print(f"Error: '{INPUT_DIR}' is not a directory.")
    else:
        # Start batch extraction
        batch_extract_data(INPUT_DIR)
