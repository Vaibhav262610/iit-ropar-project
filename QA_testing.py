import json
from pathlib import Path

from ragas.llms import LangchainLLMWrapper
import json
from langchain_openai import ChatOpenAI
from pathway.xpacks.llm.vector_store import VectorStoreClient
from datasets import Dataset


# Base configuration
BASE_DIR = Path("./finance")
QA_DATASET = BASE_DIR / "tatdqa_dataset.json"  # or .jsonl

# *naming convention -> dataset-embedder-parser-[splitter]-[extras].json
# in this file the actual question-answer pair (from tatdqa is stored), it is sent to the LLM with retrieved info and response is stored
# This json has question (from dataset), answer (actual answer from the dataset [aka expected answer]), retrieved_contexts (top k context retrieved from vector store) and response (question+context -> LLM -> response)
REFINED_DATASET_QA = (
    BASE_DIR / "12345tatdqa-qa-dataset-snowflake-arctic-embed-l-unst-singleperdoc.json"
)


llm = ChatOpenAI(model="gpt-4o-mini")
evaluator_llm = LangchainLLMWrapper(llm)


# Pathway config
PATHWAY_PORT = 8765
PATHWAY_HOST = "127.0.0.1"

client = VectorStoreClient(
    host=PATHWAY_HOST,
    port=PATHWAY_PORT,
)


def load_dataset(file_path: Path) -> list:
    """Load data from either JSON or JSONL file"""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    # Determine file type and load accordingly
    if file_path.suffix == ".jsonl":
        print(f"Loading JSONL file: {file_path}")
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        # Save as JSON for future use
        json_path = file_path.with_suffix(".json")
        print(f"Converting to JSON: {json_path}")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    elif file_path.suffix == ".json":
        print(f"Loading JSON file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    return data


def load_or_create_results(data: list, max_samples: int = 5) -> dict:
    """Load existing results or create new ones"""
    if REFINED_DATASET_QA.exists():
        print(f"Loading existing results from {REFINED_DATASET_QA}")
        with open(REFINED_DATASET_QA, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Generating new results...")
    metrics_dict = {
        "user_input": [],  # user questions
        "response": [],  # model responses
        "retrieved_contexts": [],  # retrieved contexts
        "ground_truth": [],  # reference answers
    }

    # ! The below loop is implemented for tatdqa dataset only, for other dataset change how we are extracting question, answer ,[evidence], [extras] from the dataset
    k = 0
    for item in data:
        if k == 51:
            break
        for question in item["questions"]:
            context = ""
            # Get RAG response for this question
            retireved_contexts = []
            docs = client(f"query: {question["question"]}", k=2)
            for doc in docs:
                context += doc["text"] + "/n/n"
                retireved_contexts.append(doc["text"])

            # Get model response using your existing prompt template
            # context_text = "\n\n".join(contexts)
            prompt = """You are a helpful chat assistant that helps answer query based on the given context.
            You will answer queries only on the basis of the following information: {context}
            Do not use outside knowledge to answer the query. If the answer is not contained in the provided information, just say that you don't know, don't try to make up an answer.
            """
            messages = [
                ("system", prompt.format(context=context)),
                ("human", question["question"]),
            ]
            # response = llm.invoke(messages).content
            try:
                response = llm.invoke(messages).content
                if response:  # Append only if we get a valid response
                    ground_truth = (
                        ", ".join(map(str, question["answer"]))
                        if isinstance(question["answer"], list)
                        else str(question["answer"])
                    )

                    # Add to metrics dictionary
                    if response.lower().count("i don't know") == 0:
                        k += 1
                        metrics_dict["user_input"].append(question["question"])
                        metrics_dict["response"].append(response)
                        metrics_dict["retrieved_contexts"].append(retireved_contexts)
                        metrics_dict["ground_truth"].append(ground_truth)
                    else:
                        # print(response)
                        print("LLM ko nhi pata")

            except Exception as e:
                print(f"Error processing question {e}")
                continue

            # Convert answer to string if it's not already

            break
        with open(REFINED_DATASET_QA, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {REFINED_DATASET_QA}")

    return metrics_dict


def evaluate_rag():
    """Main evaluation function"""
    # Ensure directories exist
    BASE_DIR.mkdir(exist_ok=True)

    # Load dataset (handles both JSON and JSONL)
    test_data = load_dataset(QA_DATASET)

    # Get or create results
    metrics_dict = load_or_create_results(test_data, -1)
    mod_dict = {
        "user_input": metrics_dict["user_input"][0:5],
        "response": metrics_dict["response"][0:5],
        "retrieved_contexts": metrics_dict["retrieved_contexts"][0:5],
        "ground_truth": metrics_dict["ground_truth"][0:5],
    }

    # Convert to Dataset for Ragas
    dataset = Dataset.from_dict(mod_dict)


if __name__ == "__main__":
    results = evaluate_rag()
    print("\nEvaluation Results:")
    # print(json.dumps(results, indent=2))
