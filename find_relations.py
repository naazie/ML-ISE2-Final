import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

# folder_path = os.path.join(os.getcwd(), "unigram")
# term_r = os.path.join(folder_path, "/kd_tree_method/term_relationships.csv")
# term_emb = os.path.join(folder_path, "/outputs/term_embeddings.npy")
# doc_emb = os.path.join(folder_path, "/outputs/document_embeddings.npy")
# doc_m = os.path.join(folder_path, "/outputs/doc_id_name_map.csv")
# terms = os.path.join(folder_path, "/outputs/terms.csv")

# Load data 
term_rel = pd.read_csv("ISE2-Final/unigrams/kd_tree_method/term_relationships.csv")
term_embeddings = np.load("ISE2-Final/unigrams/outputs/term_embeddings.npy")
doc_embeddings = np.load("ISE2-Final/unigrams/outputs/document_embeddings.npy")
terms_df = pd.read_csv("ISE2-Final/unigrams/outputs/terms.csv")
doc_map = pd.read_csv("ISE2-Final/unigrams/outputs/doc_id_name_map.csv")

#  user input 
target = input("Enter a term: ").strip().lower()

# Find terms 
related = term_rel[
    (term_rel["Term1"].str.lower() == target) | (term_rel["Term2"].str.lower() == target)
]

if related.empty:
    print(f"\n No related terms found for '{target}' in term_relationships.csv")
else:
    related["OtherTerm"] = related.apply(
        lambda row: row["Term2"] if row["Term1"].lower() == target else row["Term1"],
        axis=1
    )
    related_sorted = related[["OtherTerm", "Similarity"]].sort_values(by="Similarity", ascending=False)
    print(f"\n🔹 Terms related to '{target}':")
    print(related_sorted.to_string(index=False))

# Find related documents 
if target not in terms_df["term"].str.lower().values:
    print(f"\n '{target}' not found in terms.csv — cannot find related documents.")
else:
    term_index = terms_df.index[terms_df["term"].str.lower() == target].tolist()[0]
    term_vec = term_embeddings[term_index].reshape(1, -1)
    similarities = cosine_similarity(term_vec, doc_embeddings)[0]

    top_n = 10
    top_indices = np.argsort(similarities)[::-1][:top_n]
    related_docs = doc_map.iloc[top_indices].copy()
    related_docs["Similarity"] = similarities[top_indices]

    print(f"\nTop {top_n} documents related to '{target}':")
    print(related_docs[["Document_Name", "Similarity"]].to_string(index=False))