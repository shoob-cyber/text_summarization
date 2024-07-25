from flask import Flask, render_template, request
import nltk
import re
from collections import Counter, defaultdict
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

app = Flask(__name__)

# English summarization functions
def read_article(text):
    article = text.split(". ")
    sentences = [sentence.replace("[^a-zA-Z]", "").split(" ") for sentence in article]
    sentences.pop()
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)

def gen_sim_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

def generate_summary(text, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []
    sentences = read_article(text)
    sentence_similarity_matrix = gen_sim_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    for i in range(min(top_n, len(ranked_sentences))):
        summarize_text.append(" ".join(ranked_sentences[i][1]))
    return ". ".join(summarize_text)

# Hindi summarization functions
STOP_WORDS = set(["पर", "इनहे", "जिनहे", "और", "भी", "जो", "के", "में", "की", "है"])

def preprocess(text):
    words = text.split()
    filtered_words = [stem(word) for word in words if word not in STOP_WORDS]
    return ' '.join(filtered_words), Counter(filtered_words)

def stem(word):
    suffixes = ["ा", "ि", "ी", "ु", "ू", "े", "ै", "ो", "ौं", "ें"]
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

def generate_summary_hindi(text, num_sentences=5):
    preprocessed_text, word_freq = preprocess(text)
    sentences = re.split(r'[।?!]', preprocessed_text)
    sentence_scores = defaultdict(float)
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            if word in word_freq:
                sentence_scores[sentence] += word_freq[word]
        if words:
            sentence_scores[sentence] /= len(words)  # Normalize by sentence length
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    return '। '.join(top_sentences) + '।'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        text = request.form['text']
        language = request.form['language']
        if language == 'english':
            summary = generate_summary(text)
        elif language == 'hindi':
            summary = generate_summary_hindi(text)
        else:
            summary = "Language not supported."
        return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
