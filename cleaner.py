import os
import csv
import re
from collections import Counter
import multiprocessing as mp
from pypdf import PdfReader
import nltk
from nltk.corpus import stopwords
import numpy as np

nltk.download('stopwords', quiet=True)

pdf_folder = 'pdfs'
cleaned_folder = 'cleaned'

# CHANGE: Create output folder for cleaned CSVs (kept same)
os.makedirs(cleaned_folder, exist_ok=True)

# unchanged: manual_keep_set
manual_keep_set = {
    'python', 'data', 'machine', 'learning', 'neural', 'networks', 'algorithm', 'function',
    'method', 'input', 'output', 'parameters', 'process', 'training', 'accuracy', 'token',
    'contextualized', 'research', 'proposals', 'model', 'layer', 'gradient', 'time', 'language',
    'sequence', 'dataset', 'parameter', 'tensor', 'class', 'information', 'state', 'problem',
    'vector', 'features', 'weights', 'matrix', 'optimization', 'results', 'linear', 'random',
    'dropout', 'probability', 'domain', 'distribution', 'descent', 'processing', 'instance',
    'momentum', 'performance', 'sgd', 'convolutional', 'computer', 'decoder', 'encoder',
    'validation', 'source', 'analysis', 'bias', 'classification', 'stochastic', 'architechture',
    'pytorch', 'multivariate', 'kruskal', 'minkowski', 'gaussian', 'underfitting',
    'tokenizer', 'lemmatization', 'manhattan', 'euclidean', 'entropy', 'substitution',
    'polynomial', 'fisher', 'expectation', 'whitney', 'similarity', 'regression',
    'multicollinearity', 'volcano', 'normalization', 'standard', 'scaling', 'robust',
    'percentile', 'quantile', 'minmax', 'outlier', 'detection', 'zscore', 'iqr', 'mad',
    'statistical', 'ttest', 'ztest', 'mann', 'likelihood', 'anova', 'wallis', 'chi', 'pvalue',
    'qvalue', 'multiple', 'correction', 'mutual', 'aic', 'bic', 'bayes', 'maximization', 'mle',
    'map', 'singular', 'decomposition', 'pca', 'pls', 'lda', 'distance', 'mahalanobis',
    'jaccard', 'cosine', 'kl', 'js', 'earthmover', 'edit', 'imputation', 'mean', 'mode',
    'median', 'knn', 'forward', 'backward', 'simple', 'multiple', 'residual',
    'homoscedasticity', 'heteroscedasticity', 'r', 'adjusted', 'leverage', 'influence',
    'cooks', 'deviance', 'hat', 'decisiontree', 'vif', 'log', 'logit', 'probit', 'sigmoid',
    'crossentropy', 'perceptron', 'naivebayes', 'decision', 'impurity', 'svm', 'ann',
    'precision', 'recall', 'f1', 'confusion', 'imbalance', 'minority', 'majority', 'smote',
    'roc', 'auc', 'regularization', 'l1', 'l2', 'elasticnet', 'kfold', 'train', 'test',
    'stratified', 'hyperparameter', 'variance', 'overfitting', 'ensemble', 'voting', 'averaging',
    'bagging', 'randomforest', 'boosting', 'adaboost', 'xgboost', 'stacking', 'histogram',
    'boxplot', 'violin', 'scatterplot', 'tsne', 'arima', 'sarima', 'holtwinters', 'prophet',
    'movingaverage', 'exponential', 'z', 't', 'f', 'chi', 'bernoulli', 'binomial',
    'negativebinomial', 'poisson', 'pareto', 'uniform', 'geometric', 'hypergeometric', 'beta',
    'gamma', 'unsupervised', 'kmeans', 'dbscan', 'birch', 'hierarchical', 'neighbor', 'kd',
    'ball', 'locality', 'anomaly', 'isolation', 'lof', 'usvm', 'selforganizingmap',
    'association', 'apriori', 'marketbasket', 'markov', 'lift', 'support', 'coverage', 'loss',
    'softmax', 'mse', 'rmse', 'triplet', 'deep', 'rnn', 'cnn', 'lstm', 'gru', 'backpropagation',
    'gradientdescent', 'minibatch', 'nesterov', 'adam', 'rmsprop', 'attention', 'transformer',
    'gan', 'autoencoder', 'pretrained', 'transfer', 'finetuning', 'nlp', 'tfidf', 'bm25',
    'pmi', 'lsa', 'word2vec', 'glove', 'bert', 'pos', 'dependency', 'ner', 'stemming',
    'sentence', 'doc2vec',
}

stop_words = set(stopwords.words('english'))

# CHANGE: Simplified to extract only unigrams (no n=2,3,4)
def extract_unigrams(args):
    relpath, idx, total = args
    filepath = os.path.join(pdf_folder, relpath)
    print(f"Processing file {idx}/{total}")

    reader = PdfReader(filepath)
    # CHANGE: Simplified text extraction loop
    pdf_text = ' '.join(page.extract_text() or '' for page in reader.pages)

    # CHANGE: Only single-word tokens
    words = re.findall(r'\b[a-zA-Z]+\b', pdf_text.lower())

    # CHANGE: Keep only non-stopwords
    unigrams = [w for w in words if w not in stop_words]

    return relpath, unigrams


if __name__ == "__main__":
    filenames = []
    # unchanged: recursive walk
    for root, _, files in os.walk(pdf_folder):
        for fname in files:
            if fname.lower().endswith('.pdf'):
                relpath = os.path.relpath(os.path.join(root, fname), pdf_folder)
                filenames.append(relpath)

    total_files = len(filenames)
    print(f"Total PDF files found: {total_files}\n")

    # CHANGE: Pass args directly to simplified unigram extractor
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(extract_unigrams, [(fname, i + 1, total_files) for i, fname in enumerate(filenames)])

    # CHANGE: Flattened data structure (no dict per n)
    all_unigrams = []
    results_dict = {}
    for fname, unigrams in results:
        results_dict[fname] = unigrams
        all_unigrams.extend(unigrams)

    # CHANGE: Only one Counter (no per-n)
    unigram_counts = Counter(all_unigrams)

    if not unigram_counts:
        print("No unigrams found across all PDFs.")
        exit()

    # unchanged: filtering logic
    freqs = np.array(list(unigram_counts.values()))
    percentile_25 = int(np.percentile(freqs, 25))
    min_freq = max(1, percentile_25)

    must_keep = {w for w in unigram_counts.keys() if w in manual_keep_set}
    final_unigrams = {w for w, c in unigram_counts.items() if c >= min_freq} | must_keep

    # CHANGE: Simplified single summary
    print(f"Total unique unigrams={len(unigram_counts)} | kept={len(final_unigrams)} | min_freq={min_freq}")

    # CHANGE: Sorted unigrams only
    final_sorted = sorted(final_unigrams, key=lambda x: (-unigram_counts.get(x, 0), x))

    # CHANGE: Single output file (data_1gram.csv)
    output_path = os.path.join(cleaned_folder, 'data_1gram.csv')

    # unchanged: CSV writing structure
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'cleaned_text'])
        for fname in filenames:
            tokens_in_file = [t for t in final_sorted if t in results_dict[fname]]

            # unchanged: remove duplicates
            seen = set()
            tokens_unique = []
            for t in tokens_in_file:
                if t not in seen:
                    seen.add(t)
                    tokens_unique.append(t)

            cleaned_text = ' ; '.join(tokens_unique)
            writer.writerow([fname, cleaned_text])