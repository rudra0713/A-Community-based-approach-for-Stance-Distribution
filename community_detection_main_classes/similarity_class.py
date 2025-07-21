import sys
sys.path.insert(1, '/scratch/rrs99/Stance_Distribution/')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict
import pickle, nltk, torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from similarity_score_codes.return_bert_sim_score_2 import compute_sim_score


class Similarity_Class:
    def __init__(self, sentences, graph):
        self.sentences = sentences
        self.sentences_tok_lem = []
        self.sentences_tok_lem_str = []
        self.glove_dim = 300
        self.graph = graph
        self.sent_topic_words_jaccard = []
        self.sent_jac_threshold = 0.5
        # self.sent_sim_threshold = 0.63
        self.sent_sim_threshold = 0.7
        self.preprocess_sentences()

    def preprocess_sentences(self):
        stopwords = set(nltk.corpus.stopwords.words('english'))
        tag_map = defaultdict(lambda: wn.NOUN)
        # tag_map['J'] = wn.ADJ
        # tag_map['R'] = wn.ADV
        tag_map['V'] = wn.VERB
        tag_map['N'] = wn.NOUN

        lmtzr = WordNetLemmatizer()

        for sent in self.sentences:
            tokens = word_tokenize(text=sent)
            upd_sent = []
            for token, tag in pos_tag(tokens):
                # If stopwords are removed before, then output of pos_tag changes
                if tag[0] in tag_map and token not in stopwords:
                    lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
                    # print(token, "=>", lemma)
                    if lemma.lower() not in upd_sent:
                        upd_sent.append(lemma.lower())
            # print(sent, upd_sent)
            self.sentences_tok_lem.append(upd_sent)
            self.sentences_tok_lem_str.append(' '.join(upd_sent))

    def jaccard_similarity(self, x, y):
        """ returns the jaccard similarity between two lists """
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        try:
            if intersection_cardinality >= 2:
                return 0.7  # randomly chosen based on threshold of 0.5
            else:
                return 0.1
            # return intersection_cardinality / float(union_cardinality)
        except:
            return 0

    def compute_jaccard(self):
        print("Jaccard similarity")
        for i in range(len(self.sentences_tok_lem)):
            jac_sim = []
            for j in range(len(self.sentences_tok_lem)):
                jac_sim.append(self.jaccard_similarity(self.sentences_tok_lem[i], self.sentences_tok_lem[j]))
            self.sent_topic_words_jaccard.append(jac_sim)
        print(len(self.sent_topic_words_jaccard), len(self.sent_topic_words_jaccard[0]))
        sent_jac_sim_between_sentences_full = list(np.argwhere(np.array(self.sent_topic_words_jaccard) > self.sent_jac_threshold))
        sent_jac_sim_between_sentences = []
        for (i, j) in sent_jac_sim_between_sentences_full:
            # no self loops and no same topic edge between (i, j) and (j, i)
            if i != j and (j, i) not in sent_jac_sim_between_sentences:
                sent_jac_sim_between_sentences.append((i, j))
        print(sent_jac_sim_between_sentences)

        for i, j in sent_jac_sim_between_sentences:
            print("adding jac sim edges between: ", self.sentences[i], '--', self.sentences_tok_lem[i], '--',
                  self.sentences[j], '--', self.sentences_tok_lem[j])
        print()
        return sent_jac_sim_between_sentences

    def compute_sent_sim(self):
        # print("glove similarity")
        # sent_topic_words_embedding = []
        # glove_model = pickle.load(open("/scratch/rrs99/glove.6B/glove_dict_" + str(self.glove_dim) + ".p", "rb"))
        # for i, sent in enumerate(self.sentences_tok_lem):
        #     feature_vec = np.zeros((self.glove_dim,), dtype='float32')
        #     n_words = 0
        #
        #     for word in sent:
        #         if word in glove_model:
        #             n_words += 1
        #             feature_vec = np.add(feature_vec, glove_model[word])
        #     # print("number of words ", n_words)
        #     if n_words > 0:
        #         feature_vec = np.divide(feature_vec, n_words)
        #     sent_topic_words_embedding.append(feature_vec)
        # sent_topic_words_embedding = self.compute_embedding(None, self.sentences)
        sent_topic_words_embedding = compute_sim_score(None, self.sentences, compute_max=False, return_embedding=True)
        if torch.cuda.is_available():
            # pushing tensor to cpu first
            sent_topic_words_embedding = sent_topic_words_embedding.cpu().numpy()
        cos_sim = cosine_similarity(sent_topic_words_embedding)
        print("cos_sim info", len(cos_sim), len(cos_sim[0]))
        sent_embed_sim_between_sentences_full = list(np.argwhere(cos_sim >= self.sent_sim_threshold))
        sent_embed_sim_between_sentences = []
        for (i, j) in sent_embed_sim_between_sentences_full:
            # no self loops and no same topic edge between (i, j) and (j, i)
            if i != j and (j, i) not in sent_embed_sim_between_sentences and not self.graph.has_edge(i, j) and\
                    not self.graph.has_edge(j, i):
                sent_embed_sim_between_sentences.append((i, j, str(cos_sim[i][j])))
        print(sent_embed_sim_between_sentences)

        for i, j, sim in sent_embed_sim_between_sentences:
            print("adding embed sim edges between: ", i, self.sentences[i], '--', j, self.sentences[j], '--', sim)
        return sent_embed_sim_between_sentences

    # def compute_embedding(self, claim, article, compute_max=True):
    #     embedder = SentenceTransformer('stsb-mpnet-base-v2')
    #
    #     corpus_embeddings = embedder.encode(article, convert_to_tensor=True)
    #     return corpus_embeddings


if __name__ == '__main__':
    # l_o_s = ['I am a good person.', 'I live in Vancouver']
    l_o_s = ['This whole \'pro-life\' bullshit is mostly coming from religious fanatics who have been born and raised to repress women.',
             'I am not religious, but who has the right to take a life?']
    c_ob = Similarity_Class(sentences=l_o_s)
    sent_topic_sim_edges = c_ob.compute_sent_sim()
    print("sent_topic_sim_edges: ", sent_topic_sim_edges)


