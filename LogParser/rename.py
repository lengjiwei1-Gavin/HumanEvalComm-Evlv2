import json
from pathlib import Path
from collections import OrderedDict


def rename_field_in_jsonl(input_file, output_file, old_field, new_field):

    input_path = Path(input_file)
    output_path = Path(output_file)

    modified_records = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:

            record = json.loads(line, object_pairs_hook=OrderedDict)

            new_record = OrderedDict()
            for key, value in record.items():
                if key == old_field:
                    new_record[new_field] = value
                else:
                    new_record[key] = value

            modified_records.append(new_record)

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in modified_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    rename_field_in_jsonl(
        input_file="SeletcedData/selected_dataset_HumanEvalComm_model_gpt-3.5-turbo-0125_topn_1_temperature_1.0.jsonl",
        output_file="../EvalInput/60_HumanEvalComm_model_gpt-3.5-turbo-0125_topn_1_temperature_1.0.jsonl",
        old_field="Clarifying Questions",
        new_field="First Model Response"
    )