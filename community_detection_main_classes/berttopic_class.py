import wikipedia, pickle, sys
from bertopic import BERTopic
from nltk import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import umap
import nltk
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from googlesearch import search
import newspaper
from collections import defaultdict
from nltk.corpus import stopwords


class Berttopic_methods:
    def __init__(self):
        self.vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
        self.berttopic_model = None
        self.glove_dim = 300
        self.topic_sim_threshold = 0.90
        self.topic_words = {}
        self.topic_embedding = {}
        self.n_neighbors = 15
        self.n_components = 5
        self.min_topic_size = 10
        self.outlier_threshold = 0.3
        # self.outlier_threshold = 0.5

    def collect_google_articles(self, claim):
        urls = []
        # urls = list(search(claim, sleep_interval=20, num_results=200))
        print(f"trying to collect articles for the claim: {claim.lower().replace('_', ' ')}")
        for url in search(claim.lower().replace('_', ' '), sleep_interval=20, num_results=150):
            urls.append(url)
            if len(urls) % 10 == 0:
                print("len of urls: ", len(urls))

        print("all urls collected .. ", urls[:5])
        google_documents = []
        for i, url in enumerate(urls):
            try:
                article = newspaper.Article(url)
                article.download()
                article.parse()
                # content = article.text
                google_documents.append(article.text)
                # if i < 2:
                #     # print(article.text)
                #     print(article.keywords)
            except Exception as e:
                print("error: ", e)
        #     if len(contents) % 10 == 0:
        #         print("url index and length of contents: ", i, len(contents))
        print("len of google documents: ", len(google_documents))
        pickle.dump(google_documents, open('google_documents_' + claim.lower().replace(' ', '_') + '.p', 'wb'))
        return google_documents

    def collect_wiki_articles(self, claim):

        result = wikipedia.search(claim.lower().replace('_', ' '), results=500)

        # printing the result
        print("result: ", result)

        wiki_documents = [wikipedia.page(res, auto_suggest=False).content for res in result]

        print("len of wiki documents: ", len(wiki_documents))
        return wiki_documents

    def train_berttopic_with_articles(self, claim, topic_word):
        umap_model = umap.UMAP(n_neighbors=self.n_neighbors, n_components=self.n_components,
                          min_dist=0.0, metric='cosine', random_state=42)
        self.berttopic_model = BERTopic(verbose=True, embedding_model='paraphrase-MiniLM-L3-v2', vectorizer_model = self.vectorizer_model,
                                        calculate_probabilities=True, min_topic_size=self.min_topic_size, umap_model=umap_model)
        # collected_documents = self.collect_wiki_articles(claim) + self.collect_google_articles(claim)
        print(f"claim: {claim}, {claim.lower().replace(' ','_')}")
        print(f"topic word:{topic_word}")
        print('...........................')
        try:
            print("trying to load documents ... ")
            collected_documents = pickle.load(open('google_documents_' + claim.lower().replace(' ','_') + '.p', 'rb'))
        except:
            print("google documents for this claim not stored yet.")
            collected_documents = self.collect_google_articles(claim)
        all_documents = []
        try:
            stopwords.words('english')
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt')
            # Get the list of English stop words
        stop_words = set(stopwords.words('english'))
        filtered_topic_words_list = []
        if len(topic_word) > 0:
            topic_words_list = topic_word.replace('_', ' ').split()
            filtered_topic_words_list = [word for word in topic_words_list if word.lower() not in stop_words]

        for content in collected_documents:
            sentences = sent_tokenize(text=content)
            for i in range(0, len(sentences), 5):
                article = ' '.join(sentences[i: i + 5])
                art_lower = word_tokenize(text=article.lower())
                if len(filtered_topic_words_list) > 0:
                    # if topic_word in art_lower:
                    #     all_documents.append(article)
                    set1 = set(word.lower() for word in art_lower)
                    set2 = set(word.lower() for word in filtered_topic_words_list)
                    matching_words = set1.intersection(set2)
                    if matching_words:
                        all_documents.append(article)
                    else:
                        print(f"set1: {set1}")
                        print(f"set2: {set2}")

                else:
                    all_documents.append(article)

                # else:
                #     if random.random() < 0.2:
                #         print("filtered out: ", article)
        print("first document: ", all_documents[0])
        print("second document: ", all_documents[1])
        print("total documents: ", len(all_documents))

        self.berttopic_model.fit(all_documents)
        self.berttopic_model.save('berttopic_model_google_' + claim.lower().replace(' ','_'))

    def compute_topic_embeddings(self, claim, topic_word='abortion'):
        try:
            self.berttopic_model = BERTopic.load('berttopic_model_google_' + claim.lower().replace(' ','_'))
            # print("bert topic model for this claim is already stored")
        except:
            print("creating bert topic model from scratch")

            self.train_berttopic_with_articles(claim, topic_word)

        topics = self.berttopic_model.get_topics()
        self.topic_words = {topic_id: [v[0] for v in topics[topic_id]] for topic_id in topics}
        if -1 not in self.topic_words:
            self.topic_words[-1] = []  # including an outlier class to keep things uniform
        print("topic word details .. ")
        for topic_id in self.topic_words:
            print(topic_id, self.topic_words[topic_id])
        print("topic embeddings ... ", type(self.berttopic_model.topic_embeddings_))
        print(len(self.berttopic_model.topic_embeddings_), len(self.berttopic_model.topic_embeddings_[0]))
        # for key in self.topic_words:
        #     self.topic_embedding[key] = self.berttopic_model.topic_embeddings_[key + 1]  # counting for -1
        # glove_model = pickle.load(open("/scratch/rrs99/glove.6B/glove_dict_" + str(self.glove_dim) + ".p", "rb"))
        #
        # for topic_id in self.topic_words:
        #     feature_vec = np.zeros((self.glove_dim,), dtype='float32')
        #     n_words = 0
        #
        #     for word in self.topic_words[topic_id]:
        #         if word in glove_model:
        #             n_words += 1
        #             feature_vec = np.add(feature_vec, glove_model[word])
        #     # print("number of words ", n_words)
        #     if n_words > 0:
        #         feature_vec = np.divide(feature_vec, n_words)
        #     self.topic_embedding[topic_id] = feature_vec


    # def test_berttopic(self, sentences, ae_input_dim, corpus_embeddings):
    #     self.sent_topic_embeddings = np.zeros((len(sentences), self.glove_dim), dtype='float32')
    #     for i, sent in enumerate(sentences):
    #         test_topic, test_prob = self.berttopic_model.transform([sent])
    #         print("sentence: ", sent, "test_topic: ", self.topic_words[test_topic[0]])
    #         if test_topic[0] == -1:
    #             topic_vec = np.zeros((self.glove_dim,), dtype='float32')
    #         else:
    #             topic_vec = self.topic_embedding[test_topic[0]]
    #         self.sent_topic_embeddings[i] = topic_vec
    #     ae_input_dim += self.glove_dim
    #
    #     print("self.ae input dimension: ", ae_input_dim)
    #     print("sent_topic_embeddings shape: ", self.sent_topic_embeddings.shape)
    #     print("len of embedding before: ", corpus_embeddings.shape)
    #
    #     corpus_embeddings = torch.Tensor(np.append(corpus_embeddings, self.sent_topic_embeddings, axis=1))
    #
    #     print("len of embedding after: ", corpus_embeddings.shape)
    #     return ae_input_dim, corpus_embeddings

    # def test_berttopic_to_add_edges(self, sentences):
    #     self.sent_topic_embeddings = np.zeros((len(sentences), self.glove_dim), dtype='float32')
    #     test_topics, test_probs = self.berttopic_model.transform(sentences)
    #     print("test topics: ", test_topics)
    #     # test topics:  [-1, -1, -1, 4, -1, -1, 20, -1, 13, 39, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 39, -1, -1, -1, -1, -1, -1, -1]
    #     print("test probs: ", test_probs)
    #
    #     topic_sentence_mapping = {i: {'topic': v} for i, v in enumerate(test_topics)}
    #     # print(test_topics)
    #     print("topic_sentence_mapping: ", topic_sentence_mapping)
    #     # {0: {'topic': -1}, 1: {'topic': -1}, 2: {'topic': -1}, 3: {'topic': 4}, 4: {'topic': -1}, 5: {'topic': -1}, 6: {'topic': 20}, 7: {'topic': -1}, 8: {'topic': 13}, 9: {'topic': 39}, 10: {'topic': -1}, 11: {'topic': -1}, 12: {'topic': -1}, 13: {'topic': -1}, 14: {'topic': -1}, 15: {'topic': -1}, 16: {'topic': -1}, 17: {'topic': -1}, 18: {'topic': -1}, 19: {'topic': -1}, 20: {'topic': -1}, 21: {'topic': -1}, 22: {'topic': -1}, 23: {'topic': 39}, 24: {'topic': -1}, 25: {'topic': -1}, 26: {'topic': -1}, 27: {'topic': -1}, 28: {'topic': -1}, 29: {'topic': -1}, 30: {'topic': -1}}
    #     matrix_for_cos_sim = []
    #     matrix_for_cos_sim_mapping = {}
    #     # print(test_prob)
    #     # sys.exit(0)
    #     for i, test_topic in enumerate(test_topics):
    #         if test_topic == -1:
    #             topic_vec = np.zeros((self.glove_dim,), dtype='float32')
    #         else:
    #             topic_vec = self.topic_embedding[test_topic]
    #             matrix_for_cos_sim_mapping[len(matrix_for_cos_sim)] = i
    #             matrix_for_cos_sim.append(topic_vec)
    #         self.sent_topic_embeddings[i] = topic_vec
    #     print("matrix_for_cos_sim_mapping: ", matrix_for_cos_sim_mapping)
    #     # matrix_for_cos_sim_mapping: {0: 3, 1: 6, 2: 8, 3: 9, 4: 23}
    #
    #     cos_sim = cosine_similarity(matrix_for_cos_sim) # should contain pairwise similarity between each argument that is matched to a topic
    #     # print("matrix for cos sim: ")
    #     # print(cos_sim)
    #     topic_sim_between_sentences_full = list(np.argwhere(cos_sim > self.topic_sim_threshold))
    #     topic_sim_between_sentences = []
    #     for (i, j) in topic_sim_between_sentences_full:
    #         # no self loops and no same topic edge between (i, j) and (j, i)
    #         if i != j and (j, i) not in topic_sim_between_sentences:
    #             topic_sim_between_sentences.append((i, j))
    #     print("topic_sim_between_sentences: ", topic_sim_between_sentences)
    #     # topic_sim_between_sentences: [(3, 4)]
    #
    #     mapped_topic_sim_between_sentences = [(matrix_for_cos_sim_mapping[i], matrix_for_cos_sim_mapping[j]) for (i, j) in topic_sim_between_sentences]
    #     print("mapped_topic_sim_between_sentences: ", mapped_topic_sim_between_sentences)
    #     print("sentence list: ")
    #     for i, sent in enumerate(sentences):
    #         print(i, ':', sent)
    #     for i, j in mapped_topic_sim_between_sentences:
    #         print("adding topic edges between: ", sentences[i], sentences[j])
    #     # mapped_topic_sim_between_sentences: [(9, 23)]
    #     return topic_sentence_mapping, mapped_topic_sim_between_sentences


    def test_berttopic_to_add_edges_updated(self, sentences):
        # self.sent_topic_embeddings = np.zeros((len(sentences), len(self.topic_embedding[0])), dtype='float32')

        test_topics_2d = []
        test_probs_2d = np.zeros((len(sentences), len(self.topic_words) - 1))
        for id_t in range(5):
            test_topics_run, test_probs = self.berttopic_model.transform(sentences)
            print("test_topics_run: ", test_topics_run)
            test_topics_2d.append(test_topics_run)
            test_probs_2d = np.add(test_probs_2d, test_probs)
        test_probs = test_probs_2d / 5
        test_topics = [Counter(col).most_common(1)[0][0] for col in zip(*test_topics_2d)]
        print("test topics: ", test_topics)
        print("test probs: ", test_probs)
        np.save('test_topics_probs', test_probs)
        print("any argument predicted with a topic with less than 70% confidence will be treated as outlier")
        for i in range(len(test_topics)):
            if test_topics[i] != -1 and test_probs[i][test_topics[i]] < 0.7:
                test_topics[i] = -1

        # test_topics = self.berttopic_model.reduce_outliers(sentences, test_topics, strategy="embeddings", threshold=self.outlier_threshold)
        print("updated test topics: ", test_topics)
        print("sentence list: ")
        for i, sent in enumerate(sentences):
            # print(i, ':', sent, 'matched topic: ', self.topic_words[test_topics[i]], test_probs[i][test_topics[i]])
            print(i, ':', sent, 'matched topic: ', self.topic_words[test_topics[i]])

        per_topic_sent_id_ob = defaultdict(list)
        for i, v in enumerate(test_topics):
            per_topic_sent_id_ob[v].append(i)

            # if v not in per_topic_sent_id_ob:
            #     per_topic_sent_id_ob[v] = [i]
            # else:
            #     per_topic_sent_id_ob[v].append(i)
        print("per_topic_sent_id_ob details")
        for topic_id in per_topic_sent_id_ob:
            print(f"topic_id: {topic_id}, sentences: {per_topic_sent_id_ob[topic_id]}")
        sentence_to_topic_mapping = {i: {'topic': v} for i, v in enumerate(test_topics)}
        # print(test_topics)
        print("sentence_to_topic_mapping: ", sentence_to_topic_mapping)
        # {0: {'topic': -1}, 1: {'topic': -1}, 2: {'topic': -1}, 3: {'topic': 4}, 4: {'topic': -1}, 5: {'topic': -1}, 6: {'topic': 20}, 7: {'topic': -1}, 8: {'topic': 13}, 9: {'topic': 39}, 10: {'topic': -1}, 11: {'topic': -1}, 12: {'topic': -1}, 13: {'topic': -1}, 14: {'topic': -1}, 15: {'topic': -1}, 16: {'topic': -1}, 17: {'topic': -1}, 18: {'topic': -1}, 19: {'topic': -1}, 20: {'topic': -1}, 21: {'topic': -1}, 22: {'topic': -1}, 23: {'topic': 39}, 24: {'topic': -1}, 25: {'topic': -1}, 26: {'topic': -1}, 27: {'topic': -1}, 28: {'topic': -1}, 29: {'topic': -1}, 30: {'topic': -1}}
        # matrix_for_cos_sim = []
        #
        # for i, test_topic in enumerate(test_topics):
        #     topic_vec = self.topic_embedding[test_topic]
        #     matrix_for_cos_sim.append(topic_vec)
        #     self.sent_topic_embeddings[i] = topic_vec
        #
        # cos_sim = cosine_similarity(matrix_for_cos_sim) # should contain pairwise similarity between each argument that is matched to a topic
        # topic_sim_between_sentences_full = list(np.argwhere(cos_sim >= self.topic_sim_threshold))
        # topic_sim_between_sentences = []
        # for (i, j) in topic_sim_between_sentences_full:
        #     # no self loops and no same topic edge between (i, j) and (j, i)
        #     if i != j and (j, i) not in topic_sim_between_sentences and not (test_topics[i] == -1 or test_topics[j] == -1):
        #         topic_sim_between_sentences.append((i, j))
        #         if test_topics[i] != test_topics[j]:
        #             print(f"highly similar topics: {test_topics[i]}-- {test_topics[j]}")
        topic_sim_between_sentences = []

        for topic_id in per_topic_sent_id_ob:
            if topic_id != -1:
                print(f"adding edges for topic_id: {topic_id}")
                for i in range(len(per_topic_sent_id_ob[topic_id]) - 1):
                    for j in range(i + 1, len(per_topic_sent_id_ob[topic_id])):
                        sent_i = per_topic_sent_id_ob[topic_id][i]
                        sent_j = per_topic_sent_id_ob[topic_id][j]

                        topic_sim_between_sentences.append((sent_i, sent_j, str(self.topic_words[topic_id])))

        print("topic_sim_between_sentences: ", topic_sim_between_sentences)

        for i, j, _ in topic_sim_between_sentences:
            print("adding topic edges between: ", i,  sentences[i], '(', test_topics[i], ')', '----', j, sentences[j], '--', '(', test_topics[j], ')')
        # mapped_topic_sim_between_sentences: [(9, 23)]
        return sentence_to_topic_mapping, topic_sim_between_sentences

