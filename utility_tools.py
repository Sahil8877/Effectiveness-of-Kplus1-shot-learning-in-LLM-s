import json

# Utility calculation functions
def calculate_baseline_utility(e_q, e_q_d):
    """
    Baseline utility: (e(Q + D) - e(Q)) / e(Q)
    """
    if e_q > 0:
        utility = (e_q_d - e_q) / e_q
        return utility
    else:
        return None  # Return None to indicate invalid calculation

def calculate_augmented_utility(e_q_d_minus_i_list, e_q_d):
    """
    Augmented utility: (e(Q + D) - e(Q + D_{-i})) / e(Q + D_{-i})
    Computes the utility for each run and returns the average utility.
    """
    utilities = []
    for e_q_d_minus_i in e_q_d_minus_i_list:
        if e_q_d_minus_i > 0:
            utility = (e_q_d - e_q_d_minus_i) / e_q_d_minus_i
            utilities.append(utility)
    if utilities:
        average_utility = sum(utilities) / len(utilities)
        return average_utility, utilities
    else:
        return None, []

# Main utility computation
def compute_utilities(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)
    
    results = {}

    for qid, query_data in data.items():
        results[qid] = {}
        augmented_utilities = []
        
        # Extract e(Q,A) and e(Q+D,A) scores
        e_q_data = query_data.get('e(Q,A)', {})
        e_q_d_data = query_data.get('e(Q+D,A)', {})
        
        e_q_scores_dict = e_q_data.get('scores', {})
        e_q_d_scores_dict = e_q_d_data.get('scores', {})
        
        # Extract single evaluation scores
        e_q = e_q_scores_dict.get('f1_avg', -1)
        e_q_d = e_q_d_scores_dict.get('f1_avg', -1)
        
        # Calculate Baseline Utility once
        baseline_utility = calculate_baseline_utility(e_q, e_q_d)
        results[qid]['Baseline_Utility'] = baseline_utility
        
        # Process each pivot document
        for doc_id, conditions in query_data.items():
            if doc_id in ['e(Q,A)', 'e(Q+D,A)']:
                continue  # Skip baseline entries
            
            # Extract e(Q+D-i,A) scores
            e_q_d_minus_i_data = conditions.get('e(Q+D-i,A)', {})
            e_q_d_minus_i_scores_dict = e_q_d_minus_i_data.get('scores', {})
            e_q_d_minus_i_scores = e_q_d_minus_i_scores_dict.get('f1_avg', -1)
            
            # Ensure scores are lists
            if not isinstance(e_q_d_minus_i_scores, list):
                e_q_d_minus_i_scores = [e_q_d_minus_i_scores]
            
            # Calculate Augmented Utility over multiple runs
            augmented_utility, individual_utilities = calculate_augmented_utility(e_q_d_minus_i_scores, e_q_d)
            
            # Store results
            results[qid][doc_id] = {
                "Augmented_Utility": augmented_utility,
                "Individual_Utilities": individual_utilities
            }
            
            # Collect augmented utilities for averaging
            if augmented_utility is not None:
                augmented_utilities.append(augmented_utility)
        
        # Calculate average augmented utility across all documents for this query
        if augmented_utilities:
            avg_augmented_utility = sum(augmented_utilities) / len(augmented_utilities)
        else:
            avg_augmented_utility = None
        
        # Store the average augmented utility
        results[qid]['Average_Augmented_Utility'] = avg_augmented_utility

    # Save results to output file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Utilities calculated and saved to {output_file}")

# File paths

input_file = "eval_results/random_answers_kplus1_4shot_3calls_10_0_bm25_dl_19_prompt1_eval.json"
output_file = "eval_results/random_answers_kplus1_4shot_3calls_10_0_bm25_dl_19_utility_scores.json"

# Run the utility computation
compute_utilities(input_file, output_file)
