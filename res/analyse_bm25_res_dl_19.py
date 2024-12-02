import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the BM25 results file
file_path = './res/bm25_dl_19.csv'  # Update this path if necessary
bm25_res_df = pd.read_csv(file_path)

# Compute overall statistics for BM25 scores
overall_stats = bm25_res_df['score'].describe()
print("Overall BM25 Score Statistics:")
print(overall_stats)

# Compute query-level statistics
query_stats = bm25_res_df.groupby('qid')['score'].describe()
print("\nQuery-level BM25 Score Statistics (First 5 Queries):")
print(query_stats.head())

# Visualize the overall score distribution
plt.figure(figsize=(10, 6))
sns.histplot(bm25_res_df['score'], bins=30, kde=True, color='blue')
plt.title('Overall BM25 Score Distribution')
plt.xlabel('BM25 Score')
plt.ylabel('Frequency')
plt.show()

# Visualize score distribution per query
plt.figure(figsize=(15, 6))
sns.boxplot(x='qid', y='score', data=bm25_res_df)
plt.xticks(rotation=90)
plt.title('BM25 Score Distribution Per Query')
plt.xlabel('Query ID')
plt.ylabel('BM25 Score')
plt.show()

# Count how many documents have scores exceeding various thresholds
thresholds = [20.0, 25.0, 30.0, 35.0]
threshold_counts = {t: (bm25_res_df['score'] >= t).sum() for t in thresholds}

print("\nDocument Counts Above BM25 Score Thresholds:")
for t, count in threshold_counts.items():
    print(f"Threshold {t}: {count} documents")
