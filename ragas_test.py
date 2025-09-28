import json
from pathlib import Path

from ragas.llms import LangchainLLMWrapper
import json
from ragas import evaluate
from langchain_openai import ChatOpenAI
from pathway.xpacks.llm.vector_store import VectorStoreClient
from ragas import evaluate
from datasets import Dataset
from math import ceil
import pandas as pd

import os
from math import ceil


from ragas.metrics import (
    LLMContextRecall,
    LLMContextPrecisionWithReference,
    ContextPrecision,
    Faithfulness,
    ResponseRelevancy,
)

# Base configuration
BASE_DIR = Path("./finance")
BASE_NAME = "12345tatdqa-qa-dataset-snowflake-arctic-embed-l-unst-singleperdoc"
QA_DATASET = BASE_DIR / "tatdqa_dataset.json"  # or .json
METRIC_INPUT = BASE_DIR / f"{BASE_NAME}.json"
llm = ChatOpenAI(model="gpt-3.5-turbo", max_retries=10)
evaluator_llm = LangchainLLMWrapper(llm)
PATHWAY_PORT = 8123
PATHWAY_HOST = "127.0.0.1"

client = VectorStoreClient(
    host=PATHWAY_HOST,
    port=PATHWAY_PORT,
)


def merge_batches_and_calculate_average(base_dir, base_name):
    """
    Merge all batch CSV files, calculate average values for metric fields,
    and delete individual batch files after saving the final merged CSV.

    Args:
    - base_dir: Directory where batch CSV files are stored.
    - base_name: Base name for the batch CSV files.

    The final merged file is saved as `final_{base_name}.csv` in `base_dir`.
    """
    batch_files = [
        file
        for file in os.listdir(base_dir)
        if file.startswith(f"score{base_name}_") and file.endswith(".csv")
    ]
    if not batch_files:
        print("No batch CSV files found. Exiting...")
        return

    print(f"Found {len(batch_files)} batch files. Merging...")

    # Load all batch files into a list of DataFrames
    dataframes = []
    for file in batch_files:
        file_path = os.path.join(base_dir, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Merge all DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Calculate average values for all numeric columns (metrics)
    numeric_columns = merged_df.select_dtypes(include=["float64", "int64"]).columns
    average_values = merged_df[numeric_columns].mean()

    # Append average values as a new row
    avg_row = {col: average_values[col] for col in numeric_columns}
    avg_row["user_input"] = "AVERAGE"  # Placeholder to indicate averages
    merged_df = pd.concat([merged_df, pd.DataFrame([avg_row])], ignore_index=True)

    # Save final merged DataFrame
    final_file_path = os.path.join(base_dir, f"final_{base_name}.csv")
    merged_df.to_csv(final_file_path, index=False)
    print(f"Final merged file saved at: {final_file_path}")

    # Delete individual batch files
    for file in batch_files:
        file_path = os.path.join(base_dir, file)
        os.remove(file_path)
        print(f"Deleted: {file_path}")

    print("All batch files merged and deleted successfully.")


def run_ragas(metrics_dict, batch_size=3):
    """
    Run Ragas evaluation in batches with a mechanism to skip completed batches.

    Args:
    - metrics_dict: dict containing evaluation data.
    - batch_size: int, number of entries per batch.

    Saves results as JSON and CSV files for each batch.
    """
    n = len(metrics_dict["user_input"])
    num_batches = ceil(n / batch_size)  # Determine the number of batches

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n)  # Ensure not to exceed n

        # File names for the current batch
        batch_num = i + 1
        csv_path = BASE_DIR / f"score{BASE_NAME}_{batch_num}.csv"
        json_path = BASE_DIR / f"score-{BASE_NAME}_{batch_num}.json"

        # Skip if CSV already exists
        if os.path.exists(csv_path):
            print(f"Batch {batch_num} already processed. Skipping...")
            continue

        # Prepare batch dictionary
        mod_dict = {
            "user_input": metrics_dict["user_input"][start_idx:end_idx],
            "response": metrics_dict["response"][start_idx:end_idx],
            "retrieved_contexts": metrics_dict["retrieved_contexts"][start_idx:end_idx],
            "ground_truth": metrics_dict["ground_truth"][start_idx:end_idx],
        }

        # Convert to Dataset for Ragas
        dataset = Dataset.from_dict(mod_dict)

        # Define metrics
        metrics = [
            LLMContextRecall(),
            LLMContextPrecisionWithReference(),
            ContextPrecision(),
            Faithfulness(),
            ResponseRelevancy(),
        ]

        # Run evaluation
        try:
            results = evaluate(
                llm=evaluator_llm,
                dataset=dataset,
                metrics=metrics,
            )

            # Save results
            df = results.to_pandas()
            df.to_json(json_path)
            df.to_csv(csv_path, index=False)
            print(f"Batch {batch_num}/{num_batches} processed and saved.")

        except Exception as e:
            print(f"Error in processing batch {batch_num}: {e}")
            break  # Stop execution on failure (you can modify this as needed)


def evaluate_rag():
    """Main evaluation function"""
    # Ensure directories exist
    BASE_DIR.mkdir(exist_ok=True)

    # Get or create results
    metrics_dict = {}
    if METRIC_INPUT.exists():
        print(f"Loading existing results from {METRIC_INPUT}")
        with open(METRIC_INPUT, "r", encoding="utf-8") as f:
            metrics_dict = json.load(f)

    run_ragas(metrics_dict, 3)
    merge_batches_and_calculate_average(BASE_DIR, BASE_NAME)



if __name__ == "__main__":
    results = evaluate_rag()
    print("\nEvaluation Results:")
