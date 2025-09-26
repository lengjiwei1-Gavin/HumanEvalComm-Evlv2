import pandas as pd
import os
import glob


def calculate_metrics(file_path):
    """Calculate metrics for both old (Initial) and new (Improved/New) versions"""
    df = pd.read_csv(file_path)

    total_replies = len(df)
    model_name = os.path.basename(file_path).replace('.csv', '')

    # Remove "Eval_Result_" prefix if it exists
    if model_name.startswith('Eval_Result_'):
        model_name = model_name[12:]  # Remove first 12 characters "Eval_Result_"

    # OLD VERSION METRICS (Initial)
    # 1. Old Communication Rate = "Clarifying Question" replies / total replies
    old_clarifying_count = len(df[df['Initial_Classification'] == 'Clarifying Question'])
    old_communication_rate = old_clarifying_count / total_replies

    # 2. Old Good Question Rate = score "3" "Clarifying Question" replies / total replies
    old_good_question_count = len(df[
                                      (df['Initial_Classification'] == 'Clarifying Question') &
                                      (df['Initial_Score'] == '3')
                                      ])
    old_good_question_rate = old_good_question_count / total_replies

    # NEW VERSION METRICS (New)
    # 3. New Communication Rate = "Clarifying Question" replies / total replies
    new_clarifying_count = len(df[df['New_Classification'] == 'Clarifying Question'])
    new_communication_rate = new_clarifying_count / total_replies

    # 4. New Good Question Rate = score "3" "Clarifying Question" replies / total replies
    new_good_question_count = len(df[
                                      (df['New_Classification'] == 'Clarifying Question') &
                                      (df['Improved_Final_Score'] == '3')
                                      ])
    new_good_question_rate = new_good_question_count / total_replies

    # 5. False Scoring Rate = replies classified as "Code Solution" in new evaluator
    # but were scored as "Clarifying Question" in old evaluator / total replies
    false_scoring_count = len(df[
                                  (df['New_Classification'] == 'Code Solution') &
                                  (df['Initial_Classification'] == 'Clarifying Question') &
                                  (df['Initial_Score'].isin(['1', '2', '3']))
                                  ])
    false_scoring_rate = false_scoring_count / total_replies

    return {
        'Model': model_name,
        'Total_Replies': total_replies,
        'Old_Communication_Rate': round(old_communication_rate * 100, 2),
        'Old_Good_Question_Rate': round(old_good_question_rate * 100, 2),
        'New_Communication_Rate': round(new_communication_rate * 100, 2),
        'New_Good_Question_Rate': round(new_good_question_rate * 100, 2),
        'False_Scoring_Rate': round(false_scoring_rate * 100, 2)
    }


def process_all_csv_files(directory_path, output_file='metrics_results.csv'):
    """Process all CSV files in directory and output results to CSV"""
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return

    results = []
    for csv_file in csv_files:
        try:
            result = calculate_metrics(csv_file)
            results.append(result)
            print(f"Processed: {result['Model']}")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    # Convert to DataFrame and save
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(f"Processed {len(results)} files")

    return df_results


# Usage
if __name__ == "__main__":
    # Process all CSV files in EvalResults directory
    results_df = process_all_csv_files("EvalResults")

    # Display results
    if results_df is not None:
        print("\nResults Summary:")
        print(results_df.to_string(index=False))