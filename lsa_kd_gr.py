import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from gensim import corpora, models
import nltk
import matplotlib
nltk.download('stopwords', quiet=True)

N_TOPICS = 130
SIM_THRESHOLD = 0.8
NGRAM_FOLDERS = ["unigrams"]


def ensure_dirs(base_path):
    outputs_dir = os.path.join(base_path, "outputs")
    graph_dir = os.path.join(base_path, "graph_method")
    kd_dir = os.path.join(base_path, "kd_tree_method")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(kd_dir, exist_ok=True)
    return outputs_dir, graph_dir, kd_dir

for folder in NGRAM_FOLDERS:
    folder_path = os.path.join(os.getcwd(), folder)
    cleaned_csv = os.path.join(folder_path, "cleaned.csv")
    outputs_dir, graph_dir, kd_dir = ensure_dirs(folder_path)

    print(f"\nProcessing folder: {folder}")

    # -------------------- LSA --------------------
    start_time = time.time()
    df = pd.read_csv(cleaned_csv)
    # cleaned_texts = [text.split() for text in df['cleaned_text'].astype(str).tolist()]
    cleaned_texts = [text.replace(';', '').split() for text in df['cleaned_text'].astype(str).tolist()]

    # Create mapping from Doc_ID to document name
    if 'filename' in df.columns:
        doc_names = df['filename'].astype(str).tolist()
    else:
        doc_names = [f"Doc_{i}" for i in range(len(df))]

    doc_id_name_map = pd.DataFrame({
        'Doc_ID': range(len(doc_names)),
        'Document_Name': doc_names
    })
    doc_id_name_map.to_csv(os.path.join(outputs_dir, 'doc_id_name_map.csv'), index=False)

    dictionary = corpora.Dictionary(cleaned_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.85)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in cleaned_texts]

    tfidf = models.TfidfModel(doc_term_matrix)
    corpus_tfidf = tfidf[doc_term_matrix]

    lsamodel = models.LsiModel(corpus_tfidf, num_topics=N_TOPICS, id2word=dictionary)

    topics = lsamodel.print_topics(num_topics=N_TOPICS, num_words=10)
    print(f"\nSample Topics:\n{topics[:5]}")

    # Document embeddings
    corpus_lsa = lsamodel[corpus_tfidf]
    doc_embeddings_dense = np.zeros((len(doc_term_matrix), N_TOPICS))
    for i, doc_topics in enumerate(corpus_lsa):
        for topic_id, score in doc_topics:
            doc_embeddings_dense[i, topic_id] = score
    np.save(os.path.join(outputs_dir, 'document_embeddings.npy'), doc_embeddings_dense)

    # Term embeddings
    term_embeddings_matrix = lsamodel.get_topics().T
    np.save(os.path.join(outputs_dir, 'term_embeddings.npy'), term_embeddings_matrix)
    terms = [dictionary[i] for i in dictionary.keys()]
    pd.DataFrame(terms, columns=['term']).to_csv(os.path.join(outputs_dir, 'terms.csv'), index=False)

    # Top terms per doc
    doc_top_terms = []
    for i, doc in enumerate(cleaned_texts):
        bow = dictionary.doc2bow(doc)
        tfidf_doc = tfidf[bow]
        tfidf_doc_sorted = sorted(tfidf_doc, key=lambda x: x[1], reverse=True)
        top_terms = [dictionary[id] for id, _ in tfidf_doc_sorted[:5]]
        doc_top_terms.append({'Doc_ID': i, 'Top_Terms': ', '.join(top_terms)})
    df_doc_terms = pd.DataFrame(doc_top_terms)
    doc_top_terms_csv = os.path.join(outputs_dir, 'doc_top_terms.csv')
    df_doc_terms.to_csv(doc_top_terms_csv, index=False)
    print(f"LSA embeddings and top terms saved for {folder} (Time taken: {time.time() - start_time:.2f}s)")


    doc_embeddings = doc_embeddings_dense
    term_embeddings = np.load(os.path.join(outputs_dir, "term_embeddings.npy"))

    doc_id_name_map = pd.read_csv(os.path.join(outputs_dir, "doc_id_name_map.csv"))
    terms_df = pd.read_csv(os.path.join(outputs_dir, "terms.csv"))

    # KD Tree --------------------------
    # KD tree basically is a binary tree only 
    # so each node has 2 children so what? 
    # so each level
    TOP_K_DOCS = 10
    TOP_K_TERMS = 100
    SIM_THRESHOLD = 0.6 

    nbrs_docs = NearestNeighbors(n_neighbors=TOP_K_DOCS, metric="cosine").fit(doc_embeddings)
    distances_docs, indices_docs = nbrs_docs.kneighbors(doc_embeddings)

    doc_relationships = []
    for i, neighbors in enumerate(indices_docs):
        for j in range(1, len(neighbors)):  # skip itself
            doc1_id = i
            doc2_id = neighbors[j]
            similarity = 1 - distances_docs[i][j]
            if similarity >= SIM_THRESHOLD:
                doc1_name = doc_id_name_map.loc[doc_id_name_map["Doc_ID"] == doc1_id, "Document_Name"].values[0]
                doc2_name = doc_id_name_map.loc[doc_id_name_map["Doc_ID"] == doc2_id, "Document_Name"].values[0]
                doc_relationships.append((doc1_id, doc2_id, doc1_name, doc2_name, similarity))

    df_doc_rel = pd.DataFrame(doc_relationships, columns=["Doc1_ID", "Doc2_ID", "Doc1_Name", "Doc2_Name", "Similarity"])
    df_doc_rel.to_csv(os.path.join(kd_dir, "document_relationships.csv"), index=False)

    terms_list = terms_df["term"].astype(str).tolist()
    nbrs_terms = NearestNeighbors(n_neighbors=TOP_K_TERMS, metric="cosine").fit(term_embeddings)
    distances_terms, indices_terms = nbrs_terms.kneighbors(term_embeddings)

    term_relationships = []
    for i, neighbors in enumerate(indices_terms):
        for j in range(1, len(neighbors)):  # skip itself
            term1 = terms_list[i]
            term2 = terms_list[neighbors[j]]
            similarity = 1 - distances_terms[i][j]
            if similarity >= SIM_THRESHOLD:
                term_relationships.append((term1, term2, similarity))

    df_term_rel = pd.DataFrame(term_relationships, columns=["Term1", "Term2", "Similarity"])
    df_term_rel.to_csv(os.path.join(kd_dir, "term_relationships.csv"), index=False)

    print("\nKD-Tree analysis complete")

# -------------------- 4. GRAPH ALGORITHMS --------------------
    def graph_algorithms(name, embeddings_path, labels_path, graph_dir, sim_threshold):
        labels_df = pd.read_csv(labels_path)

        if 'Document_Name' in labels_df.columns:
            labels = labels_df['Document_Name'].astype(str).tolist()
        elif 'term' in labels_df.columns:
            labels = labels_df['term'].astype(str).tolist()
        else:
            labels = labels_df.iloc[:, 0].astype(str).tolist()

        doc_embeddings_dense = np.load(embeddings_path)  
        from sklearn.metrics.pairwise import cosine_similarity
        # SIM_THRESHOLD = [0.6, 0.7, 0.8, 0.9] 
        # We can test with various threshold values
        sim_matrix = cosine_similarity(doc_embeddings_dense)

        G = nx.Graph()
        G.add_nodes_from(labels)

        row_idx, col_idx = np.where(np.triu(sim_matrix, k=1) >= sim_threshold)
        for i, j in zip(row_idx, col_idx):
            sim = sim_matrix[i, j]
            G.add_edge(labels[i], labels[j], weight=sim)

        import community as community_louvain

        partition = community_louvain.best_partition(G)
        # modularity = community_louvain.modularity(partition, G)

        community_assignments = pd.DataFrame(list(partition.items()), columns=['Label', 'Community'])
        community_assignments.to_csv(os.path.join(graph_dir,f"{name}_community_assignments.csv"), index=False)

        # Group docs by community
        community_groups = {}
        for label, comm in partition.items():
            community_groups.setdefault(comm, []).append(label)

        community_data = [(comm, docs, len(docs)) for comm, docs in community_groups.items()]

        sorted_communities = sorted(community_data, key=lambda x: x[2], reverse=True)

        community_docs_df = pd.DataFrame({
            'Community': [com for com, docs, size in sorted_communities],
            'Size': [size for com, docs, size in sorted_communities],
            'Documents': [', '.join(docs) for com, docs, size in sorted_communities]
        })

        community_docs_df.to_csv(os.path.join(graph_dir, f"{name}_communities_sorted.csv"), index=False)


        # Visualisation 
        community_colors = [partition[label] for label in G.nodes()]
        # plt.figure(figsize=(10, 8))
        # nx.draw_networkx(G, node_color=community_colors, with_labels=True, node_size=50, cmap = matplotlib.colormaps["tab20"])
        # plt.title(f"{name.title()} Communities with Labels")
        # plt.savefig(os.path.join(graph_dir, f"{name}_communities_with_labels.png"))
        # plt.show()
        # plt.close()
        plt.figure(figsize=(10, 8))
        nx.draw_networkx(G, node_color=community_colors, with_labels=False, node_size=50, cmap = matplotlib.colormaps["tab20"])
        plt.title(f"{name.title()} Communities without Labels")
        plt.savefig(os.path.join(graph_dir, f"{name}_communities_without_labels.png"))
        plt.show()
        plt.close()
        
        from pyvis.network import Network
        import random
        # Pyvis visualization
        net = Network(height='800px', width='1000px', notebook=False)

        # Generate colors for each community (random colors)
        unique_communities = set(partition.values())
        color_map = {comm: '#%02X%02X%02X' % (random.randint(0,255), random.randint(0,255), random.randint(0,255)) for comm in unique_communities}

        for node in G.nodes():
            comm = partition[node]
            label = str(node)
            net.add_node(node, label=label, color=color_map[comm])

        for u, v in G.edges():
            net.add_edge(u, v)

        pyvis_path = os.path.join(graph_dir, f"{name}_communities_pyvis.html")
        # net.show(pyvis_path, notebook= False)
        net.write_html(pyvis_path)
        print(f"Saved interactive pyvis graph to {pyvis_path}")
        
    start_time = time.time()
    graph_algorithms('document', os.path.join(outputs_dir, 'document_embeddings.npy'), os.path.join(outputs_dir, 'doc_id_name_map.csv'), graph_dir, 0.8)

    graph_algorithms('term', os.path.join(outputs_dir, 'term_embeddings.npy'), os.path.join(outputs_dir, 'terms.csv'), graph_dir, 0.95) 
    print(f"Graph algorithms completed for {folder} (Time taken: {time.time() - start_time:.2f}s)")