import pandas as pd
import pickle

class QueryDocumentPair:
    def __init__(self, qid: str, docno: str, qText: str, dText: str, qrel: int, score: float):
        self.qid = qid
        self.docno = docno
        self.qText = qText
        self.dText = dText
        self.qrel = qrel
        self.rsv = score
        self.pred = None

    def put_prediction(self, pred: float):
        self.pred = pred

    def __str__(self) -> str:
        return f'qid_{self.qid}: {self.qText}; docno_{self.docno}; qrel: {self.qrel}; bm25_rsv: {self.rsv}.'

def get_msmarco_passage_pairs(dataset: str):
    """
    Loads BM25 results, queries, and MSMARCO passage text.
    Returns a list of QueryDocumentPair objects and the document dictionary.
    """

    # Load different datasets based on version
    if 'v2' not in dataset:
        dl_p1_res_df = pd.read_csv('./res/bm25_dl_19.csv')
        dl_p2_res_df = pd.read_csv('./res/bm25_dl_20.csv')
        queries = pd.read_csv('./middle_products/queries.csv')

        # Load MSMARCO document dictionary
        with open('./middle_products/msmarco_doc_dict.pkl', 'rb') as f:
            msmarco_doc_dict = pickle.load(f)

    else:
        dl_p1_res_df = pd.read_csv('./res/bm25_dl_21.csv')
        dl_p2_res_df = pd.read_csv('./res/bm25_dl_22.csv')
        queries = pd.read_csv('./middle_products/queries_v2.csv')

        # Load MSMARCO document dictionary
        with open('./middle_products/msmarco_doc_dict_v2.pkl', 'rb') as f:
            msmarco_doc_dict = pickle.load(f)

    # Ensure that docno in dl_p1_res_df and dl_p2_res_df are strings
    dl_p1_res_df['docno'] = dl_p1_res_df['docno'].astype(str).str.strip()
    dl_p2_res_df['docno'] = dl_p2_res_df['docno'].astype(str).str.strip()

    # Ensure that docno in msmarco_doc_dict are strings
    msmarco_doc_dict = {str(k): v for k, v in msmarco_doc_dict.items()}

    print(f"Loaded MSMARCO doc dictionary with {len(msmarco_doc_dict)} documents.")

    # Debugging: Check the document IDs available in msmarco_doc_dict
    print("First 50 document IDs in MSMARCO doc dictionary:")
    print(list(msmarco_doc_dict.keys())[:50])  # Print first 50 document IDs

    q_d_pair_list = []
    dl_qids = list(set(dl_p1_res_df.qid.unique()) | set(dl_p2_res_df.qid.unique()))

    for qid in dl_qids:
        qText = queries.loc[queries.qid == qid, 'query'].values[0]
        if qid in dl_p1_res_df.qid.unique():
            df_for_qid = dl_p1_res_df[dl_p1_res_df.qid == qid]
        else:
            df_for_qid = dl_p2_res_df[dl_p2_res_df.qid == qid]
        df_for_qid = df_for_qid.sort_values('rank', ascending=True)

        for docno, score in df_for_qid[['docno', 'score']].values[:100 if qid in dl_p1_res_df.qid.unique() else 50]:
            # Ensure docno is a string and sanitized
            docno = str(docno).strip()

            # Check if docno is in the document dictionary
            if docno not in msmarco_doc_dict:
                print(f"Warning: Document {docno} not found in msmarco_doc_dict! Check if the document ID is correct.")
                dText = "Document not found"
            else:
                dText = msmarco_doc_dict[docno]

            # Set a default qrel (since we removed qrels processing)
            qrel = 0

            # Create the QueryDocumentPair object and append it
            q_d_pair = QueryDocumentPair(qid=str(qid), docno=docno, qText=qText, dText=dText, qrel=qrel, score=float(score))
            q_d_pair_list.append(q_d_pair)

    return q_d_pair_list, msmarco_doc_dict

def compose_context_kplus1(qid, query, k_plus_one_examples, main_context, msmarco_doc_dict, batch_size, max_tokens=300):
    """
    Composes the prompt for the K+1 experiment based on the given examples.

    Parameters:
    - qid: Query ID.
    - query: The query text.
    - k_plus_one_examples: List of document examples to include in the context.
    - main_context: Additional main context if any (not used in this version).
    - msmarco_doc_dict: Dictionary mapping docno to document text.
    - batch_size: Number of documents to include (for compatibility).
    - max_tokens: Maximum tokens allowed in the context.

    Returns:
    - A string containing the composed prompt.
    """

    # Initialize the context
    context_sections = []

    # Add each document to the context
    token_count = 0
    
    for example in k_plus_one_examples:
        # Ensure 'docno' is available
        if 'docno' not in example:
            print(f"Missing 'docno' in example: {example}")
            continue  # Skip this example if 'docno' is missing

        docno = str(example['docno']).strip()
        example_text = msmarco_doc_dict.get(docno, "Document not found")

        if example_text == "Document not found":
            print(f"Warning: Document {docno} not found in msmarco_doc_dict.")
            continue  # Skip this document if not found

        example_tokens = len(example_text.split())
        if token_count + example_tokens <= max_tokens:
            context_sections.append(f"{example_text}")
            token_count += example_tokens
        else:
            break  # Stop adding documents if max_tokens exceeded

    # Combine all context sections
    context_text = "\n".join(context_sections)

    # Construct the final prompt
    if context_text:
        # There are context documents
        
        preamble = (
            "You are an expert tasked with answering questions based on your knowledge and the provided contexts.\n"
        )
        context = "Contexts :"
        prompt = f"{preamble}\n{context}\n{context_text}\n\nQuestion :\n{query}?\n"
    else:
        # No context documents
        preamble = (
            "You are an expert tasked with answering questions based on your knowledge.\n\n"
        )
        prompt = f"{preamble}\nQuestion :\n{query}?\n"

    return prompt
