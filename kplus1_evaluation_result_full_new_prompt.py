import pandas as pd
import pickle
import json
import sys
from evaluate import load
import logging

# Configure logging
logging.basicConfig(
    filename='./debug_logs/evaluation.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Set to DEBUG for detailed logs
)
logger = logging.getLogger(__name__)

def prepare_qids_qrels_docdict(dataset_name):
    """Load MSMARCO qids, qrels, and document dictionary."""
    try:
        # Load the document dictionary
        doc_dict_path = './middle_products/msmarco_passage_v1_retrieved_top_tail.pkl'
        logger.debug(f"Loading document dictionary from {doc_dict_path}")
        with open(doc_dict_path, 'rb') as f:
            doc_dict = pickle.load(f)
        logger.info("Document dictionary loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading MSMARCO doc dictionary: {e}")
        sys.exit(1)
    
    try:
        # Load the queries
        queries_path = f'./middle_products/queries_{dataset_name}.csv'
        logger.debug(f"Loading queries from {queries_path}")
        queries = pd.read_csv(queries_path)
        queries['qid'] = queries['qid'].astype('str')
        qids = queries.qid.tolist()
        logger.info(f"Queries loaded successfully. Total queries: {len(qids)}")
    except Exception as e:
        logger.error(f"Error loading queries: {e}")
        sys.exit(1)
    
    try:
        # Load the qrels with the correct separator and column names
        qrels_path = f'./middle_products/qrels.csv'
        logger.debug(f"Loading qrels from {qrels_path}")
        qrels = pd.read_csv(qrels_path, sep=',', names=['qid', 'docno', 'label', 'iteration'], header=0)
        qrels['qid'] = qrels['qid'].astype('str')
        qrels['docno'] = qrels['docno'].astype('str')
        qrels['label'] = qrels['label'].astype(int)
        logger.info(f"Qrels loaded successfully. Total qrels: {len(qrels)}")
    except Exception as e:
        logger.error(f"Error loading qrels: {e}")
        sys.exit(1)
    
    return qids, qrels, doc_dict

def get_references(qid, doc_dict, qrels):
    """Get relevant document texts for a given query ID."""
    relevant_texts = []
    labels_used = []

    # Attempt to get documents with label 3
    relevant_qrels_label3 = qrels[(qrels.qid == qid) & (qrels.label == 3)]
    if not relevant_qrels_label3.empty:
        labels_used.append(3)
        relevant_docnos_label3 = relevant_qrels_label3.docno.tolist()
        logger.debug(f"QID {qid} - Relevant DocNos with label 3: {relevant_docnos_label3}")
        
        # Retrieve document texts for label 3 documents
        missing_docs_label3 = []
        for docno in relevant_docnos_label3:
            doc_text = doc_dict.get(docno)
            if doc_text:
                relevant_texts.append(doc_text)
            else:
                missing_docs_label3.append(docno)
        if missing_docs_label3:
            logger.warning(f"QID {qid} - Missing label 3 documents in doc_dict: {missing_docs_label3}")
    else:
        logger.info(f"No qrels with label 3 found for QID {qid}.")

    # If no texts retrieved from label 3 documents, attempt to get label 2 documents
    if not relevant_texts:
        logger.info(f"Trying to get qrels with label 2 for QID {qid}.")
        relevant_qrels_label2 = qrels[(qrels.qid == qid) & (qrels.label == 2)]
        if not relevant_qrels_label2.empty:
            labels_used.append(2)
            relevant_docnos_label2 = relevant_qrels_label2.docno.tolist()
            logger.debug(f"QID {qid} - Relevant DocNos with label 2: {relevant_docnos_label2}")
            
            # Retrieve document texts for label 2 documents
            missing_docs_label2 = []
            for docno in relevant_docnos_label2:
                doc_text = doc_dict.get(docno)
                if doc_text:
                    relevant_texts.append(doc_text)
                else:
                    missing_docs_label2.append(docno)
            if missing_docs_label2:
                logger.warning(f"QID {qid} - Missing label 2 documents in doc_dict: {missing_docs_label2}")
        else:
            logger.warning(f"No qrels with label 2 found for QID {qid}.")

    if not relevant_texts:
        logger.warning(f"No relevant documents available in doc_dict for QID {qid}")
        return [], labels_used

    logger.debug(f"QID {qid} - Retrieved {len(relevant_texts)} document texts with labels {labels_used}")
    return relevant_texts, labels_used


def evaluator(answer, references, qid, bertscore):
    """Evaluate a single answer against the references."""
    if not references:
        logger.warning(f"No references available for QID {qid}. Returning -1 for all scores.")
        return {
            "precision_avg": -1,
            "precision_max": -1,
            "recall_avg": -1,
            "recall_max": -1,
            "f1_avg": -1,
            "f1_max": -1,
        }
    
    predictions = [answer] * len(references)
    try:
        logger.debug(f"Computing BERTScore for QID {qid}")
        results = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            model_type="bert-base-uncased",
            verbose=False,
            rescale_with_baseline=True
        )
    except Exception as e:
        logger.error(f"Error computing BERTScore for QID {qid}: {e}")
        return {
            "precision_avg": -1,
            "precision_max": -1,
            "recall_avg": -1,
            "recall_max": -1,
            "f1_avg": -1,
            "f1_max": -1,
        }
    
    precision = results["precision"]
    recall = results["recall"]
    f1 = results["f1"]
    
    return {
        "precision_avg": sum(precision) / len(precision),
        "precision_max": max(precision),
        "recall_avg": sum(recall) / len(recall),
        "recall_max": max(recall),
        "f1_avg": sum(f1) / len(f1),
        "f1_max": max(f1),
    }

def extract_answer_from_calls(calls_dict):
    """Extracts the answer from the model's calls."""
    for call_index in sorted(calls_dict.keys(), key=int):
        result = calls_dict[call_index]
        if result:
            if isinstance(result, dict) and 'answer' in result:
                return result['answer']
            elif isinstance(result, str):
                return result
    return ""  # If all calls are empty

if __name__ == "__main__":
    # Parse arguments
    if len(sys.argv) < 6:
        print("Usage: python evaluate_script.py batch_size num_calls dataset_name top_num tail_num suffix [retriever_name]")
        sys.exit(1)
    batch_size = int(sys.argv[1])
    num_calls = int(sys.argv[2])
    dataset_name = sys.argv[3]
    top_num = sys.argv[4]
    tail_num = sys.argv[5]
    suffix = "_prompt1"
    retriever_name = sys.argv[7] if len(sys.argv) > 7 else "bm25"
    
    file_path = f'./middle_products/random_answers_kplus1_{batch_size}shot_{num_calls}calls_{top_num}_{tail_num}_{retriever_name}_dl_{dataset_name}{suffix}.json'
    eval_file_path = f'./eval_results/random_answers_kplus1_{batch_size}shot_{num_calls}calls_{top_num}_{tail_num}_{retriever_name}_dl_{dataset_name}{suffix}_eval.json'
    
    # Load BERTScore
    logger.info("Loading BERTScore")
    bertscore = load("bertscore")
    
    # Prepare MSMARCO data
    qids, qrels, doc_dict = prepare_qids_qrels_docdict(dataset_name)
    
    # Load generated answers
    try:
        logger.debug(f"Loading generated answers from {file_path}")
        with open(file_path, "r") as f:
            answer_book = json.load(f)
        logger.info("Generated answers loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading answers: {e}")
        sys.exit(1)
    
    # Initialize evaluation results
    results = {}
    
    # Evaluate each query
    for qid in answer_book:
        logger.debug(f"Evaluating QID {qid}")
        results[qid] = {}
        
        # Get references for the current query
        references, labels_used = get_references(qid, doc_dict, qrels)
        logger.debug(f"Number of references for QID {qid}: {len(references)} (Labels used: {labels_used})")

        if not references:
            logger.warning(f"No references available for QID {qid}. Skipping evaluation for this query.")
            continue  # Skip to the next query

        # Evaluate e(Q,A)
        e_QA_calls = answer_book[qid].get('e(Q,A)', {})
        e_QA_response = extract_answer_from_calls(e_QA_calls)
        if not e_QA_response:
            logger.warning(f"No answer found for e(Q,A) for QID {qid}")
        e_QA_scores = evaluator(e_QA_response, references, qid, bertscore)
        results[qid]['e(Q,A)'] = {
            'answer': e_QA_response,
            'scores': e_QA_scores,
            'label_used': labels_used
        }
        
        # Evaluate e(Q+D,A)
        e_QDA_calls = answer_book[qid].get('e(Q+D,A)', {})
        e_QDA_response = extract_answer_from_calls(e_QDA_calls)
        if not e_QDA_response:
            logger.warning(f"No answer found for e(Q+D,A) for QID {qid}")
        e_QDA_scores = evaluator(e_QDA_response, references, qid, bertscore)
        results[qid]['e(Q+D,A)'] = {
            'answer': e_QDA_response,
            'scores': e_QDA_scores,
            'label_used': labels_used
        }
        
        # Evaluate e(Q+D-i,A) for each pivot document
        for pivot_docno in answer_book[qid]:
            if pivot_docno in ['e(Q,A)', 'e(Q+D,A)']:
                continue
            results[qid][pivot_docno] = {}
            e_QDminusiA_calls = answer_book[qid][pivot_docno].get('e(Q+D-i,A)', {})
            e_QDminusiA_response = extract_answer_from_calls(e_QDminusiA_calls)
            if not e_QDminusiA_response:
                logger.warning(f"No answer found for e(Q+D-i,A) for QID {qid}, DocNo {pivot_docno}")
            e_QDminusiA_scores = evaluator(e_QDminusiA_response, references, qid, bertscore)
            results[qid][pivot_docno]['e(Q+D-i,A)'] = {
                'answer': e_QDminusiA_response,
                'scores': e_QDminusiA_scores,
                'label_used': labels_used
            }
    
    # Save results
    try:
        with open(eval_file_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Evaluation completed. Results saved to {eval_file_path}.")
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")
