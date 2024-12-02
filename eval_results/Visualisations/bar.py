import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON file with utility scores
with open("eval_results/random_answers_kplus1_4shot_1calls_10_0_bm25_dl_19_utility_scores.json", "r") as f:
    data = json.load(f)

# Prepare data for visualization
queries = []
baseline_utilities = []
augmented_utilities = []

for qid, docs in data.items():
    for doc_id, scores in docs.items():
        queries.append(f"Q{qid}_D{doc_id}")
        baseline_utilities.append(scores["baseline_utility"])
        augmented_utilities.append(scores["augmented_utility"])

# Plot a bar chart
x = np.arange(len(queries))
width = 0.4

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, baseline_utilities, width, label="Baseline Utility", color='skyblue')
plt.bar(x + width/2, augmented_utilities, width, label="Augmented Utility", color='orange')

# Customize the plot
plt.xlabel("Query-Document Pairs", fontsize=12)
plt.ylabel("Utility Scores", fontsize=12)
plt.title("Baseline vs Augmented Utility", fontsize=14)
plt.xticks(x, queries, rotation=90, fontsize=10)
plt.legend()
plt.tight_layout()
plt.show()
