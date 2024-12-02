from permutation_generator import *

def used_preamble():
    return "You are an expert at answering questions based on your own knowledge and related context. Please answer this question based on the given context. End your answer with STOP."

def used_preamble_0():
    return "You are an expert at answering questions based on your own knowledge. Please answer this question. End your answer with STOP."

# prepare needed files
def prepare_data(dataset_name: str, retriever_name = 'bm25'):
    import pandas as pd
    # read the retrieved documents
    import pickle
    
    if(retriever_name == 'bm25'):
        with open('./middle_products/msmarco_passage_v1_retrieved_top_tail.pkl', 'rb') as f:
            doc_dict = pickle.load(f)
            f.close()
    elif('oracle' in retriever_name):
        with open('./middle_products/msmarco_passage_v1_qrels.pkl', 'rb') as f:
            doc_dict = pickle.load(f)
            f.close()
    elif(retriever_name == 'mt5'):
        with open('./middle_products/msmarco_passage_v1_retrieved_mt5.pkl', 'rb') as f:
            doc_dict = pickle.load(f)
            f.close()
    else:
        print('this retriever is not supported')
        return
    # prepare queries
    queries = pd.read_csv(f'./middle_products/queries_{dataset_name}.csv')
    # prepare res file
    res = pd.read_csv(f'./res/{retriever_name}_dl_{dataset_name}.csv') # retrieval result
      
    return doc_dict, queries, res

# compose the examples in the context part
import pandas as pd
from utility_tools import calculate_relevance_utility, compute_effectiveness_delta

def compose_context_with_utilities(res, qid, batch_size, batch_step, top_starts, tail_starts, doc_dict):
    """
    Compose contexts for varying start ranks, including utility and delta calculations.

    Args:
        res (pd.DataFrame): Query-document retrieval results.
        qid (int): Query ID.
        batch_size (int): Number of documents per context batch.
        batch_step (int): Step size for moving through document ranks.
        top_starts (int): Number of top-ranked starts to include.
        tail_starts (int): Number of tail-ranked starts to include.
        doc_dict (dict): Document ID to text mapping.

    Returns:
        tuple: Start ranks, contexts, utilities, and deltas.
    """
    retrieved_for_q = res[res.qid == qid]
    retrieved_num = retrieved_for_q['rank'].max() + 1

    starts = list(range(0, (retrieved_num - 1) - (batch_size - 1) + 1, batch_step))
    start_rank_list = list(set(starts[:top_starts]).union(set(starts[-tail_starts:])))
    start_rank_list.sort()

    context_book = []
    utilities = []
    deltas = []  # For effectiveness improvement

    for i, start in enumerate(start_rank_list):
        context = ''
        end = start + batch_size
        batch_docnos = retrieved_for_q[(retrieved_for_q['rank'] >= start) & (retrieved_for_q['rank'] < end)].docno.tolist()
        batch_texts = [doc_dict[str(docno)] for docno in batch_docnos]
        batch_relevance_scores = retrieved_for_q[(retrieved_for_q['rank'] >= start) & (retrieved_for_q['rank'] < end)].relevance.tolist()

        # Create context and calculate utility
        num = 0
        for text in batch_texts:
            num += 1
            context += f'Context {num}: "{text}";\n'

        u_k = calculate_relevance_utility(batch_relevance_scores)
        context_book.append(context)
        utilities.append(u_k)

        # Calculate delta effectiveness if there's a previous utility
        if i > 0:
            deltas.append(compute_effectiveness_delta(utilities[i - 1], u_k))
        else:
            deltas.append(None)  # No delta for the first batch

    return start_rank_list, context_book, utilities, deltas


def compose_context_with_permutations(res, qid: str, batch_size, batch_step, top_starts, tail_starts, doc_dict, full_permutations):
    
    print(qid)
    retrieved_for_q = res[res.qid==qid]
    retrieved_num = retrieved_for_q['rank'].max()+1
      
    starts = list(range(0, (retrieved_num-1)-(batch_size-1)+1, batch_step))
    start_rank_list = list(set(starts[:top_starts]).union(set(starts[(len(starts)-1)-(tail_starts-1):])))
    print(start_rank_list)
    start_rank_list.sort()
      
    p_name_list = []
    context_book = []
    for start in start_rank_list:
        end = start + batch_size
        batch_docnos = retrieved_for_q[(retrieved_for_q['rank']>=start)&(retrieved_for_q['rank']<end)].docno.tolist()

        permuntation_docnos = get_permutation(batch_docnos, len(batch_docnos), full_permutations=full_permutations)
            
        for p_name, p_batch_docnos in permuntation_docnos.items():
            context = ''
                  
            batch_texts = [doc_dict[str(docno)] for docno in p_batch_docnos]
            num = 0
            for text in batch_texts:
                num += 1
                context += f'Context {num}: "{text}";\n'
                  
            p_name_list.append(f'{start}>{p_name}')
            context_book.append(context)
            
    return p_name_list, context_book