import json
import csv
from pathlib import Path


def process_jsonl_file(file_path):
    csv_path = file_path.with_suffix(".csv")

    with open(file_path, "r", encoding="utf-8") as f_in, open(csv_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            ["Index", "ID", "Initial_Score", "Improved_Final_Score", "Initial_Classification", "New_Classification"])

        for idx, line in enumerate(f_in, start=1):
            try:
                data = json.loads(line.strip())

                item_id = data.get("ID")

                initial_score = data.get("Evaluator Quality Score", None)
                if isinstance(initial_score, str) and initial_score.strip() == "":
                    initial_score = "Null"  # 转换成 null

                improved_score = None
                new_classification = None
                if isinstance(data.get("Improved_Evaluation_Result"), dict):
                    improved_result = data["Improved_Evaluation_Result"]

                    improved_score = improved_result.get("final_score", None)
                    if isinstance(improved_score, str) and improved_score.strip() == "":
                        improved_score = "Null"

                    new_classification = improved_result.get("classification", None)
                    if isinstance(new_classification, str) and new_classification.strip() == "":
                        new_classification = "Null"


                initial_classification = None


                if new_classification == "Empty Response":
                    initial_classification = "Empty Response"
                else:

                    if initial_score in ["1", "2", "3"]:
                        initial_classification = "Clarifying Question"
                    else:
                        initial_classification = "Code Solution"

                writer.writerow(
                    [idx, item_id, initial_score, improved_score, initial_classification, new_classification])

            except Exception as e:
                print(f"⚠️ Error in line {idx} of {file_path.name}: {e}")


def batch_process(eval_dir="EvalResults"):
    path = Path(eval_dir)
    for file in path.glob("*.jsonl"):
        print(f"Processing {file.name} ...")
        process_jsonl_file(file)
        print(f"✅ Saved {file.stem}.csv")


if __name__ == "__main__":
    batch_process("EvalResults")