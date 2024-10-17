import argparse
import csv
import json
import requests
from bert_score import score
import pandas as pd
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate LLM answer quality using BERTScore in parallel.')

    # Input and output file paths
    parser.add_argument('--input_csv', type=str, default='questions.csv', help='Path to the input CSV file containing questions and reference answers. Default: questions.csv')
    parser.add_argument('--output_csv', type=str, default='report.csv', help='Path to the output CSV file to save the evaluation report. Default: report.csv')

    # Optional parameters for model, temperature, and max tokens
    parser.add_argument('--vllm_url', type=str, default='http://localhost:8000/v1/chat/completions', help='URL of the vLLM server API endpoint. Default: http://localhost:8000/v1/chat/completions')
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.3', help='The model name to use for inference. Default: mistralai/Mistral-7B-Instruct-v0.3')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling during inference. Default: 0.7')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum number of tokens to generate. Default: 100')

    # Optional number of workers
    parser.add_argument('--max_workers', type=int, default=10, help='Maximum number of parallel threads for sending requests. Default: 10')

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

def process_question(vllm_url, model_name, temperature, max_tokens, row):
    question_id = row['question_id']
    question_text = row['question_text']
    reference_answer = row['reference_answer']
    generated_answer = send_question_to_vllm(vllm_url, model_name, temperature, max_tokens, question_text)
    return {
        'question_id': question_id,
        'question_text': question_text,
        'generated_answer': generated_answer,
        'reference_answer': reference_answer
    }

def main():
    args = parse_arguments()

    # Read input CSV
    df_input = read_input_csv(args.input_csv)

    report_data = []

    print("Starting batch processing of questions with parallel requests...\n")

    # Use ThreadPoolExecutor for parallel requests
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Create a list of futures
        futures = {executor.submit(process_question, args.vllm_url, args.model_name, args.temperature, args.max_tokens, row): index for index, row in df_input.iterrows()}
        # Use tqdm to display progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Questions"):
            result = future.result()
            report_data.append(result)

    print("\nCompleted batch processing. Evaluating answers with BERTScore...\n")

    # Extract generated and reference answers
    generated_answers = [entry['generated_answer'] for entry in report_data]
    reference_answers = [entry['reference_answer'] for entry in report_data]

    # Evaluate using BERTScore
    P, R, F1 = evaluate_answers(generated_answers, reference_answers)

    # Add BERTScore metrics to report data
    for i in range(len(report_data)):
        report_data[i]['precision'] = round(P[i], 4)
        report_data[i]['recall'] = round(R[i], 4)
        report_data[i]['f1'] = round(F1[i], 4)

    # Write to output CSV
    write_output_csv(args.output_csv, report_data)

if __name__ == "__main__":
    main()
