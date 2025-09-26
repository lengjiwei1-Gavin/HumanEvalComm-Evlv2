# HumanEvalComm-LLMEval V2: An Enhanced Evaluation Framework for LLM Communication Competence

This project is a reproduction, validation, and enhancement of the benchmark introduced in **HumanEvalComm: Benchmarking the Communication Competence of Code Generation for LLMs and LLM Agent**.

We systematically address the "False Communication" and "Scoring Bias" issues present in the original evaluator by introducing an independent Classification Module and an innovative Multi-Level LLM Jury scoring system. The result is a more accurate, objective, and traceable pipeline for evaluating the communication competence of LLMs.

## Features

- **Modular Design**: Core functionalities are split into independent modules, such as the classifier (`LLMBasedClas.py`) and the rater (`ReliableRater.py`), facilitating isolated testing and maintenance.

- **Separation of Config and Logic**: All API keys, model names, and endpoints are managed in `config.json`, keeping them separate from the core application logic for better security and easier configuration.

- **Engineered Prompts**: All prompt templates are centralized in `prompts.py`, allowing for easy iteration and optimization of prompts without altering the main program flow.

- **End-to-End Data Pipeline**: Provides a complete set of scripts for the entire workflow, from parsing raw logs to final results analysis.

