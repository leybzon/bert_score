import argparse
import csv
import json
import requests
from bert_score import score
import pandas as pd
import sys
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate LLM answer quality using BERTScore.')

    # Input and output file paths
    parser.add_argument('--input_csv', type=str, default='questions.csv', help='Path to the input CSV file containing questions and reference answers. Default: questions.csv')
    parser.add_argument('--output_csv', type=str, default='report.csv', help='Path to the output CSV file to save the evaluation report. Default: report.csv')

    # Optional parameters for model, temperature, and max tokens
    parser.add_argument('--vllm_url', type=str, default='http://localhost:8000/v1/chat/completions', help='URL of the vLLM server API endpoint. Default: http://localhost:8000/v1/chat/completions')
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.3', help='The model name to use for inference. Default: mistralai/Mistral-7B-Instruct-v0.3')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling during inference. Default: 0.7')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum number of tokens to generate. Default: 100')

    return parser.parse_args()

def read_input_csv(input_csv_path):
    try:
        df = pd.read_csv(input_csv_path)
        required_columns = {'question_id', 'question_text', 'reference_answer'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Input CSV is missing required columns: {missing}")
        return df
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        sys.exit(1)

def send_question_to_vllm(vllm_url, model_name, temperature, max_tokens, question_text):
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question_text}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(vllm_url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            completion = response.json()
            answer = completion['choices'][0]['message']['content'].strip()
            return answer
        else:
            print(f"Error: Received status code {response.status_code} for question: {question_text}")
            return "Error: Failed to generate answer."
    except Exception as e:
        print(f"Exception while sending request to vLLM: {e}")
        return "Error: Exception occurred."

def evaluate_answers(generated_answers, reference_answers):
    # Calculate BERTScore
    P, R, F1 = score(generated_answers, reference_answers, lang="en", verbose=True)
    # Convert tensors to lists
    P = P.tolist()
    R = R.tolist()
    F1 = F1.tolist()
    return P, R, F1

def write_output_csv(output_csv_path, report_data):
    try:
        df_report = pd.DataFrame(report_data)
        df_report.to_csv(output_csv_path, index=False)
        print(f"Report successfully saved to {output_csv_path}")
    except Exception as e:
        print(f"Error writing output CSV: {e}")
        sys.exit(1)

def main():
    args = parse_arguments()

    # Read input CSV
    df_input = read_input_csv(args.input_csv)

    generated_answers = []
    reference_answers = df_input['reference_answer'].tolist()

    print("Starting batch processing of questions...\n")

    for index, row in df_input.iterrows():
        question_id = row['question_id']
        question_text = row['question_text']
        print(f"Processing Question ID {question_id}: {question_text}")
        generated_answer = send_question_to_vllm(args.vllm_url, args.model_name, args.temperature, args.max_tokens, question_text)
        generated_answers.append(generated_answer)
        # To avoid overwhelming the server, add a short delay
        time.sleep(0.1)

    print("\nCompleted batch processing. Evaluating answers with BERTScore...\n")

    # Evaluate using BERTScore
    P, R, F1 = evaluate_answers(generated_answers, reference_answers)

    # Prepare report data
    report_data = []
    for i in range(len(df_input)):
        report_entry = {
            'question_id': df_input.at[i, 'question_id'],
            'question_text': df_input.at[i, 'question_text'],
            'generated_answer': generated_answers[i],
            'reference_answer': reference_answers[i],
            'precision': round(P[i], 4),
            'recall': round(R[i], 4),
            'f1': round(F1[i], 4)
        }
        report_data.append(report_entry)

    # Write to output CSV
    write_output_csv(args.output_csv, report_data)

if __name__ == "__main__":
    main()
