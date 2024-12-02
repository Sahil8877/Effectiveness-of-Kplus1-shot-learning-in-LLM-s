from llama_cpp import Llama
from compose_prompts_kplus1 import compose_context_kplus1  # Adjusted import
from llama_tools import llama_tools
from experiment_tools import *
from prompt1_tools import *
import json
import sys
import os
import pandas as pd
import pickle
import logging

# Configure logging
logging.basicConfig(
    filename='./debug_logs/experiment.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Set to DEBUG for detailed logs
)
logger = logging.getLogger(__name__)

def select_top_k_plus_one_documents(dl_res_df, qid, k_plus_one_size, msmarco_doc_dict):
    """
    Selects the top k+1 documents for a given query based solely on BM25 scores.
    Returns a list of document dictionaries containing 'docno', 'qrel', and 'rsv'.
    """
    required_dl_columns = ['qid', 'docno', 'score']

    # Ensure the necessary columns are present in dl_res_df
    if not all(col in dl_res_df.columns for col in required_dl_columns):
        raise ValueError(f"dl_res_df is missing required columns: {required_dl_columns}")

    # Ensure docno is a string and strip whitespace
    dl_res_df['docno'] = dl_res_df['docno'].astype(str).str.strip()

    query_results = dl_res_df[dl_res_df['qid'] == qid]

    if query_results.empty:
        logger.warning(f"No BM25 results found for QID {qid}.")
        return []

    # Sort by BM25 score in descending order
    sorted_results = query_results.sort_values(by='score', ascending=False)

    # Select top k+1 documents
    top_k_plus_one = sorted_results.head(k_plus_one_size)

    # Convert to list of dicts with required fields
    documents = top_k_plus_one[['docno', 'score']].to_dict(orient='records')
    processed_documents = [{'docno': doc['docno'], 'qrel': 0, 'rsv': doc['score']} for doc in documents]

    logger.debug(f"Selected top {k_plus_one_size} documents for QID {qid}: {processed_documents}")

    return processed_documents

def store_results_to_json(results, output_file):
    """
    Store the generated results in a well-structured JSON file.
    Uses atomic write to prevent data corruption.
    """
    temp_file = output_file + ".tmp"
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        os.replace(temp_file, output_file)
        logger.info(f"Results successfully saved to {output_file}.")
    except IOError as e:
        logger.error(f"Failed to write results to {output_file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("This experiment takes 8 parameters: ")
        print("1. batch size")
        print("2. batch step")
        print("3. number of calls")
        print("4. top starts")
        print("5. tail starts")
        print("6. temperature")
        print("7. dataset version (e.g., 19/20)")
        print("8. retriever name (optional, default: bm25)")
        print("Example: 1 1 1 10 0 0.3 19 reverse_oracle")
        sys.exit()

    # Parse input parameters
    try:
        batch_size = int(sys.argv[1])        # k
        batch_step = int(sys.argv[2])
        num_calls = int(sys.argv[3])
        top_starts = int(sys.argv[4])
        tail_starts = int(sys.argv[5])
        temperature = float(sys.argv[6])
        dataset_name = str(sys.argv[7])
        retriever_name = 'bm25'
        if len(sys.argv) == 9:
            retriever_name = str(sys.argv[8])
    except ValueError as e:
        logger.error(f"Invalid input parameters: {e}")
        sys.exit(1)

    # Load the LLM and data
    try:
        llm = llama_tools.load_llama()
        logger.info("LLaMA model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load LLaMA model: {e}")
        sys.exit(1)

    try:
        doc_dict, queries, res = prepare_data(dataset_name, retriever_name)
        logger.info(f"Data prepared successfully for dataset {dataset_name} with retriever {retriever_name}.")
    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        sys.exit(1)

    # File paths for saving settings and results
    setting_file_name = f'./middle_products/random_answers_kplus1_{batch_size}shot_{num_calls}calls_{top_starts}_{tail_starts}_{retriever_name}_dl_{dataset_name}_prompt1_settings.json'
    file_name = f'./middle_products/random_answers_kplus1_{batch_size}shot_{num_calls}calls_{top_starts}_{tail_starts}_{retriever_name}_dl_{dataset_name}_prompt1.json'

    # Save settings
    setting_record = {
        'batch_size': batch_size,
        'batch_step': batch_step,
        'num_calls': num_calls,
        'top_starts': top_starts,
        'tail_starts': tail_starts,
        'temperature': temperature,
        'dataset_name': dataset_name,
        'retriever_name': retriever_name
    }
    try:
        with open(setting_file_name, "w+", encoding='UTF-8') as f:
            json.dump(setting_record, f, indent=4)
        logger.info(f"Settings saved to {setting_file_name}.")
    except IOError as e:
        logger.error(f"Failed to write settings to {setting_file_name}: {e}")
        sys.exit(1)

    # Load or initialize results
    try:
        with open(file_name, "r") as f:
            result_to_write = json.load(f)
            existed_qids_list = list(result_to_write.keys())
        logger.info(f"Loaded existing results from {file_name}. Total QIDs: {len(existed_qids_list)}")
    except FileNotFoundError:
        result_to_write = {}
        existed_qids_list = []
        logger.info("No existing results found. Starting fresh.")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for {file_name}: {e}")
        result_to_write = {}
        existed_qids_list = []
    except Exception as e:
        logger.error(f"Error loading existing results: {e}")
        result_to_write = {}
        existed_qids_list = []

    # Debug folder setup
    debug_folder = "./debug_logs"
    os.makedirs(debug_folder, exist_ok=True)
    logger.info(f"Debug logs will be saved to {debug_folder}.")

    # Load BM25 results once before the main loop
    bm25_path = f'./res/bm25_dl_{dataset_name}.csv'  # Updated to use dataset_name
    if not os.path.isfile(bm25_path):
        logger.error(f"BM25 results file does not exist at path: {bm25_path}")
        sys.exit(1)

    try:
        dl_p1_res_df = pd.read_csv(bm25_path)
        logger.info(f"BM25 results loaded successfully with shape: {dl_p1_res_df.shape}")
    except FileNotFoundError as e:
        logger.error(f"BM25 results file not found: {e}")
        sys.exit(1)
    except pd.errors.EmptyDataError as e:
        logger.error(f"BM25 results file is empty: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading BM25 results: {e}")
        sys.exit(1)

    # Load MSMARCO doc dict
    msmarco_doc_dict_path = './middle_products/msmarco_passage_v1_retrieved_top_tail.pkl'
    try:
        with open(msmarco_doc_dict_path, 'rb') as f:
            msmarco_doc_dict = pickle.load(f)
        logger.info("MSMARCO document dictionary loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"MSMARCO doc dict file not found: {e}")
        sys.exit(1)
    except pickle.UnpicklingError as e:
        logger.error(f"Error unpickling MSMARCO doc dict: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading MSMARCO doc dict: {e}")
        sys.exit(1)

    q_no = 0
    # Iterate over queries
    for qid, query in zip(queries['qid'].tolist(), queries['query'].tolist()):
        logger.info(f'q_number={q_no}--{qid}')
        q_no += 1

        # Skip already processed queries
        if str(qid) in existed_qids_list:
            logger.info(f"QID {qid} already processed. Skipping.")
            continue
        else:
            logger.info(f"Processing QID {qid}.")

        # Fetch query-document pairs and select k+1 examples
        try:
            k_plus_one_size = batch_size + 1
            k_plus_one_examples = select_top_k_plus_one_documents(
                dl_res_df=dl_p1_res_df,
                qid=qid,
                k_plus_one_size=k_plus_one_size,
                msmarco_doc_dict=msmarco_doc_dict
            )
            if not k_plus_one_examples:
                logger.warning(f"No documents found for QID {qid}. Skipping.")
                continue
        except Exception as e:
            logger.error(f"Error selecting k+1 documents for QID {qid}: {e}")
            continue

        # Initialize results for this query
        result_to_write[qid] = {}

        # Generate e(Q,A): Response with query only
        try:
            llm.set_seed(1000)
            logger.info(f'Generating e(Q,A) for QID {qid}')

            # Prepare prompt with query only
            prompt_e_QA = compose_context_kplus1(
                qid=qid,
                query=query,
                k_plus_one_examples=[],  # No documents
                main_context="",  # Or appropriate context
                msmarco_doc_dict=msmarco_doc_dict,
                batch_size=0,  # No documents
                max_tokens=300
            )
            logger.debug(f"Prompt for e(Q,A):\n{prompt_e_QA}")

            # Call LLaMA for e(Q,A)
            multi_call_e_QA = {}
            for j in range(num_calls):
                result = llama_tools.single_call(llm=llm, prompt=prompt_e_QA, temperature=temperature)
                if not result:
                    logger.warning(f"Empty result for e(Q,A) for QID {qid}, call {j}")
                multi_call_e_QA[j] = result
            # Store e(Q,A)
            result_to_write[qid]['e(Q,A)'] = multi_call_e_QA
        except Exception as e:
            logger.error(f"Error generating e(Q,A) for QID {qid}: {e}")
            continue

        # Generate e(Q+D,A): Response with full set of k+1 documents
        try:
            llm.set_seed(1000)
            logger.info(f'Generating e(Q+D,A) for QID {qid}')

            # Prepare prompt with query and full set of k+1 documents
            prompt_e_QDA = compose_context_kplus1(
                qid=qid,
                query=query,
                k_plus_one_examples=k_plus_one_examples,  # All k+1 documents
                main_context="",  # Or appropriate context
                msmarco_doc_dict=msmarco_doc_dict,
                batch_size=batch_size + 1,  # k+1 documents
                max_tokens=300
            )
            logger.debug(f"Prompt for e(Q+D,A):\n{prompt_e_QDA}")

            # Call LLaMA for e(Q+D,A)
            multi_call_e_QDA = {}
            for j in range(num_calls):
                result = llama_tools.single_call(llm=llm, prompt=prompt_e_QDA, temperature=temperature)
                if not result:
                    logger.warning(f"Empty result for e(Q+D,A) for QID {qid}, call {j}")
                multi_call_e_QDA[j] = result
            # Store e(Q+D,A)
            result_to_write[qid]['e(Q+D,A)'] = multi_call_e_QDA
        except Exception as e:
            logger.error(f"Error generating e(Q+D,A) for QID {qid}: {e}")
            continue

        # For each pivot document
        for pivot_idx, pivot_doc in enumerate(k_plus_one_examples):
            logger.info(f"\tPivot {pivot_idx + 1}/{k_plus_one_size}: DocNo {pivot_doc['docno']}")

            # Exclude the pivot to form D-i
            D_minus_i = [doc for idx, doc in enumerate(k_plus_one_examples) if idx != pivot_idx]

            # Generate e(Q+D-i,A): Response excluding pivot document
            try:
                llm.set_seed(1000)
                logger.info(f'Generating e(Q+D-i,A) for QID {qid}, Pivot DocNo {pivot_doc["docno"]}')

                # Prepare prompt with query and D-i documents
                prompt_e_QDminusiA = compose_context_kplus1(
                    qid=qid,
                    query=query,
                    k_plus_one_examples=D_minus_i,
                    main_context="",
                    msmarco_doc_dict=msmarco_doc_dict,
                    batch_size=batch_size,  # k documents
                    max_tokens=300
                )
                logger.debug(f"Prompt for e(Q+D-i,A) for Pivot DocNo {pivot_doc['docno']}:\n{prompt_e_QDminusiA}")

                # Call LLaMA for e(Q+D-i,A)
                multi_call_e_QDminusiA = {}
                for j in range(num_calls):
                    result = llama_tools.single_call(llm=llm, prompt=prompt_e_QDminusiA, temperature=temperature)
                    if not result:
                        logger.warning(f"Empty result for e(Q+D-i,A) for QID {qid}, Pivot DocNo {pivot_doc['docno']}, call {j}")
                    multi_call_e_QDminusiA[j] = result
                # Store e(Q+D-i,A)
                result_to_write[qid][pivot_doc['docno']] = {'e(Q+D-i,A)': multi_call_e_QDminusiA}
            except Exception as e:
                logger.error(f"Error generating e(Q+D-i,A) for QID {qid}, Pivot DocNo {pivot_doc['docno']}: {e}")
                continue

            # Save results after each pivot
            store_results_to_json(result_to_write, file_name)
            logger.info(f"Results for QID {qid}, Pivot DocNo {pivot_doc['docno']} saved.")

    # Cleanup
    try:
        llm.close()
        logger.info("LLaMA model closed successfully.")
    except Exception as e:
        logger.error(f"Error closing LLaMA model: {e}")
