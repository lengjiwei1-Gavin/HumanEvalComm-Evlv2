# HumanEvalComm-LLMEval V2: An Enhanced Evaluation Framework for LLM Communication Competence
This project is a reproduction, validation, and enhancement of the benchmark introduced in **HumanEvalComm: Benchmarking the Communication Competence of Code Generation for LLMs and LLM Agent**.
We systematically address the "False Communication" and "Scoring Bias" issues present in the original evaluator by introducing an independent Classification Module and an innovative Multi-Level LLM Jury scoring system. The result is a more accurate, objective, and traceable pipeline for evaluating the communication competence of LLMs.
## Features
- **Modular Design**: Core functionalities are split into independent modules, such as the classifier (`LLMBasedClas.py`) and the rater (`ReliableRater.py`), facilitating isolated testing and maintenance.
- **Separation of Config and Logic**: All API keys, model names, and endpoints are managed in `config.json`, keeping them separate from the core application logic for better security and easier configuration.
- **Engineered Prompts**: All prompt templates are centralized in `prompts.py`, allowing for easy iteration and optimization of prompts without altering the main program flow.
- **End-to-End Data Pipeline**: Provides a complete set of scripts for the entire workflow, from parsing raw logs to final results analysis.
## Prerequisites
1. Clone this repository.
2. **Configure API Keys**: Open `config.json` and enter your API keys for OpenAI, Anthropic (Claude), and DeepSeek.
## Quick Start: Running the Evaluation Pipeline
The entire enhanced evaluation pipeline can be executed with a single command using the `main.py` script. It automatically processes all `.jsonl` dataset files found in the specified input directory.
### Command:
```bash
python main.py --input_dir <your_input_directory> --output_dir <your_output_directory> --samples <number_of_samples_per_file>
```
### Arguments:
- `--input_dir`: The folder containing your datasets in `.jsonl` format. Defaults to `EvalInput`.
- `--output_dir`: The folder where evaluation results will be saved. Defaults to `EvalResults`.
- `--samples`: The number of records to process from each file, useful for quick tests. Set to `-1` to process all records.
### Example:
To process the first 10 samples from each file in the `EvalInput` directory and save results to `EvalResults`:
```bash
python main.py --input_dir EvalInput --output_dir EvalResults --samples 10
```
## Data Preparation Workflow
Before running the main pipeline, the raw log files from the original HumanEvalComm benchmark must be processed into the required `.jsonl` format.
The `LogParser/` directory contains the necessary scripts for this preparation. The general workflow is:
1. Use `LogParser/LogSelector.py` to sample a consistent set of problems from the original raw log files (`.log_1`).
2. Use `LogParser/Extractor.py` to convert the sampled text files into the structured `.jsonl` format required by our pipeline.
3. Place the final `.jsonl` files into the `EvalInput` directory to be processed by `main.py`.
After running the evaluation, you can use `ResultCalculator.py` to aggregate the results from the `EvalResults` directory and compute the final metrics.
## Modular Design & Unit Testing
The core logic for classification and rating is designed to be run standalone for unit testing and debugging. This is achieved via the `if name == 'main':` block in each module script.
### LLMBasedClas.py
Can be run directly to test the classification module on a small, hardcoded sample of data. This allows for rapid validation of the classifier's performance and prompt effectiveness without executing the full pipeline.
### ReliableRater.py
Can be run directly to test the multi-level jury rating system. It processes a few sample records and prints the detailed voting process and final scores, which is useful for debugging the rating logic and verifying API connectivity for all models.
**Note**: Running these modules directly is intended for development and testing purposes. Configuration details, like test file paths, are hardcoded within the scripts for convenience.

