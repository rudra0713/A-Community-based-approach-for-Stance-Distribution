import sys
# sys.path.insert(1, '/scratch/rrs99/Stance_Distribution/')
import networkx as nx
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
from collections import defaultdict
import pickle
from nltk import sent_tokenize, word_tokenize
from keybert import KeyBERT
import nltk
import spacy
nltk.download('wordnet')

from nltk.corpus import wordnet as wn


class Keyword_similarity_Class:
    def __init__(self, sentences, articles, graph, topic_word, claim):
        self.sentences = sentences
        self.articles = articles
        self.kb_model = KeyBERT()
        self.keywords = []
        self.graph = graph
        self.claim = claim
        self.topic_word = topic_word
        # self.keyword_sim_threshold = 0.3  # determined after looking 200 articles on abortion
        self.keyword_sim_threshold = 0.4
        self.keyword_to_org_word_mapping = {}
        self.extract_keywords_based_on_keybert(articles)
        # initialize language model
        self.nlp = spacy.load("en_core_web_md")

        # add pipeline (declared through entry_points in setup.py)
        # In my local machine's venv, there is a file at "/Users/rudra/PycharmProjects/Stance_Distribution/sd_dist_venv/lib/python3.8/site-packages/data_spacy_entity_linker/wikidb_filtered.db
        # which was not being created in server, so I had to copy that from my machine.
        self.nlp.add_pipe("entityLinker", last=True)
        # self.nlp.add_pipe("entity_linker",  last=True)
        self.entity_ob = defaultdict(list)
        self.extract_and_link_entities()

        # self.preprocess_articles()
        # self.lemmatizer = WordNetLemmatizer()

    # def preprocess_articles(self):
    #
    #     collected_documents = pickle.load(open('google_documents_' + self.claim.lower().replace(' ','_') + '.p', 'rb'))
    #     # collected_documents = pickle.load(open('google_documents_gayrights_should_be_legalized..p', 'rb'))
    #     print(f"len of collected documents: {len(collected_documents)}")
    #     all_documents = []
    #     for content in collected_documents:
    #         sentences = sent_tokenize(text=content)
    #         for i in range(0, len(sentences), 5):
    #             article = ' '.join(sentences[i: i + 5])
    #             art_lower = word_tokenize(text=article.lower())
    #             if self.topic_word in art_lower:
    #                 all_documents.append(article)
    #     print(f"len of all documents: {len(all_documents)}")
    #     self.extract_keywords_based_on_keybert(all_documents)

    def find_full_word(self, token, sentence):
        # Use word boundary regex to find whole word containing the token
        pattern = r'\b\w*' + re.escape(token) + r'\w*\b'
        match = re.search(pattern, sentence)

        return match.group(0) if match else None

    def compute_keyword_sim(self):
        # def get_hyponyms(word):
        #     hyponyms = []
        #     synsets = wn.synsets(word)
        #     for synset in synsets:
        #         hyponyms.extend(synset.hyponyms())
        #     return get_words_from_synsets(hyponyms)
        #
        # def get_words_from_synsets(synsets):
        #     words = []
        #     print(f"synsets: {synsets}")
        #     for synset in synsets:
        #         words.extend(synset.lemma_names())
        #     return list(set(words))
        #
        # keyword_sim_edges = []
        # number_of_edges_per_word_count = defaultdict(int)
        # sentences_tokenized = [[porter_stemmer.stem(word.lower()) for word in word_tokenize(sentence)]
        #                        for sentence in self.sentences]
        # for word, conf in self.keywords:
        #     for i in range(len(self.sentences) - 1):
        #         for j in range(i + 1, len(self.sentences)):
        #             sent_i = sentences_tokenized[i]
        #             sent_j = sentences_tokenized[j]
        #             if not self.graph.has_edge(i, j) and word in sent_i and word in sent_j:
        #                 number_of_edges_per_word_count[(word, conf)] += 1
        #                 # keyword_sim_edges.append((i, j, word))
        #                 # print("adding keyword sim edges between: ", i, self.sentences[i], '--', j, self.sentences[j], '--', word)
        # print(f"number_of_edges_per_word_count: {number_of_edges_per_word_count}")
        # print("edges per keyword ... ")
        # words_to_remove = []
        # words_to_add = []
        # for key in list(number_of_edges_per_word_count):
        #     word, conf = key
        #     ratio = round(number_of_edges_per_word_count[key] / sum(number_of_edges_per_word_count.values()), 2)
        #     print(f"keyword: {word}, {ratio}")
        #     if ratio > 0.7:
        #         words_to_remove.append((word, conf))
        #         # self.keywords.remove((word, conf))
        #         hyp_words = get_hyponyms(self.keyword_to_org_word_mapping[word])
        #         for hy_word in hyp_words:
        #             words_to_add.append((porter_stemmer.stem(hy_word).lower(), conf))
        #
        #             # self.keywords.append((hy_word, conf))
        # for word, conf in words_to_remove:
        #     self.keywords.remove((word, conf))
        # for word, conf in words_to_add:
        #     self.keywords.append((word, conf))
        # pickle.dump(self.keywords, open('extracted_keywords_' + self.topic_word + '.p', 'wb'))
        # print("final self.keywords")
        # for word_info in self.keywords:
        #     print(word_info)
        # print(".......")
        # for word, conf in self.keywords:
        #     for i in range(len(self.sentences) - 1):
        #         for j in range(i + 1, len(self.sentences)):
        #             sent_i = sentences_tokenized[i]
        #             sent_j = sentences_tokenized[j]
        #             if not self.graph.has_edge(i, j) and word in sent_i and word in sent_j:
        #                 # number_of_edges_per_word_count[(word, conf)] += 1
        #                 keyword_sim_edges.append((i, j, word))
        keyword_sim_edges = []

        for word, conf in self.keywords:
            for i in range(len(self.sentences) - 1):
                for j in range(i + 1, len(self.sentences)):
                    if not self.graph.has_edge(i, j) and word in self.sentences[i] and word in self.sentences[j]:
                        matched_word_sentence_i = self.find_full_word(word, self.sentences[i])
                        matched_word_sentence_j = self.find_full_word(word, self.sentences[j])
                        if not matched_word_sentence_i or not matched_word_sentence_j or matched_word_sentence_i == matched_word_sentence_j:
                            keyword_sim_edges.append((i, j, word))
                        else:
                            try:
                                edge_label = matched_word_sentence_i + "/" + matched_word_sentence_j
                                keyword_sim_edges.append((i, j, edge_label))
                            except Exception as e:
                                print("keyword edge label error: ", matched_word_sentence_i, type(matched_word_sentence_i), matched_word_sentence_j, type(matched_word_sentence_j))
                                sys.exit(0)
                        # number_of_edges_per_word_count[(word, conf)] += 1
                        print(f"adding keyword similarity edges between: {i, self.sentences[i]} --- {j, self.sentences[j]} --- {word}")

        return keyword_sim_edges


    def extract_keywords_based_on_keybert(self, texts):
        word_imp_ob = {}
        # for text in texts:
            # print(text)
        word_imp_list_all_texts = self.kb_model.extract_keywords(texts, keyphrase_ngram_range=(1, 3), stop_words='english')
        keywords_ob = {}
        print(f"word_imp_list_all_texts: {word_imp_list_all_texts}")
        for word_imp_list_per_text in word_imp_list_all_texts:
            for word, imp in word_imp_list_per_text:
                if word in keywords_ob:
                    keywords_ob[word] = max(imp, keywords_ob[word])
                else:
                    keywords_ob[word] = imp
        print(f"keywords_ob: {keywords_ob.items()}")
        # for word_org, imp in word_imp_list_per_text:
        #     word = porter_stemmer.stem(word_org).lower()
        #     self.keyword_to_org_word_mapping[word] = word_org
        #
        #     if word not in word_imp_ob:
        #         word_imp_ob[word] = [imp]
        #     else:
        #         word_imp_ob[word].append(imp)
        # for word in word_imp_ob:
        #     word_imp_ob[word] = sum(word_imp_ob[word]) / len(word_imp_ob[word])
        # word_imp_list = []
        # for word in word_imp_ob:
        #     word_imp_list.append((word, word_imp_ob[word]))
        # self.keywords = sorted(word_imp_list, key=lambda w: w[1], reverse=True)
        # self.keywords = [(word, conf) for word, conf in word_imp_list_per_text if word not in [self.topic_word, porter_stemmer.stem(self.topic_word).lower()] and conf >= self.keyword_sim_threshold]
        print(f"topic word: {self.topic_word}")
        print(f"keyword_sim_threshold: {self.keyword_sim_threshold}")
        # self.keywords = [(word, conf) for word, conf in keywords_ob.items() if word not in [self.topic_word, porter_stemmer.stem(self.topic_word).lower()] and conf >= self.keyword_sim_threshold]
        for word, conf in keywords_ob.items():
            if word not in [self.topic_word,
                            porter_stemmer.stem(self.topic_word).lower()] and conf >= self.keyword_sim_threshold:
                self.keywords.append((word, conf))

        print("done extracting keywords: ", len(self.keywords))
        print("all keywords ... ")
        for k, c in self.keywords:
            print(k, c)
        print("..................")

    def add_entity_edges(self):
        entity_edges = []
        for entity_text in self.entity_ob:
            for i in range(len(self.entity_ob[entity_text]) - 1):
                for j in range(i + 1, len(self.entity_ob[entity_text])):
                    if not self.graph.has_edge(i, j):
                        sent_ind_i = self.entity_ob[entity_text][i]
                        sent_ind_j = self.entity_ob[entity_text][j]
                        # number_of_edges_per_word_count[(word, conf)] += 1
                        print(f"adding entity edges between: {sent_ind_i, self.sentences[sent_ind_i]} --- {sent_ind_j, self.sentences[sent_ind_j]} --- {entity_text}")
                        entity_edges.append((sent_ind_i, sent_ind_j, entity_text))
        return entity_edges

    def extract_and_link_entities(self):
        allowed_entity_types = ["EVENT", "FAC", "GPE", "LAW", "NORP", "ORG", "PERSON", "PER"]

        for i, sent in enumerate(self.sentences):
            doc = self.nlp(sent)
            main_entities = []
            # for ent in doc.ents:
            #     print(ent.text, ent.label_)  # Display the entity and its type (e.g., PERSON, ORG)
            main_entities = [ent.text for ent in doc.ents if ent.label_ in allowed_entity_types]
            main_entities = list(set(main_entities))
            for ent in main_entities:
                self.entity_ob[ent].append(i)
                # if ent in self.entity_ob:
                #     self.entity_ob[ent]['ids'] = [i]
                # else:
                #     self.entity_ob[ent] = {'ids': [i]}
        entity_linker_ob = defaultdict(list)
        for i, sent in enumerate(self.sentences):
            doc = self.nlp(sent)
            all_linked_entities = doc._.linkedEntities
            for ent in all_linked_entities:
                if ent.get_span() in self.entity_ob:
                    entity_linker_ob[ent.get_id()].append(ent.get_span())
                    # if ent.get_id() not in self.entity_ob:
                    #     self.entity_ob[ent.get_id()] = {'ids': [i], 'word': ent.get_span()}
                    # else:
                    #     self.entity_ob[ent.get_id()]['ids'].append(i)
        for ent in entity_linker_ob:
            if len(entity_linker_ob[ent]) > 1:
                print(f"going to map {entity_linker_ob[ent]} into {entity_linker_ob[ent][0]}")
                for ent_text in entity_linker_ob[ent][1:]:
                    first_text = entity_linker_ob[ent][0]
                    self.entity_ob[first_text].extend(self.entity_ob[ent_text])
        print("final list of entities")
        for entity_text in self.entity_ob:
            print(f"entity_text: {entity_text}, entity: {self.entity_ob[entity_text]}")


if __name__ == '__main__':
    # l_o_s = ['I am a good person.', 'I live in Vancouver']
    # pass
    l_o_s = ['indonesia is good', 'lanka is bad', 'lanka is good']
    g = nx.Graph()
    o = Keyword_similarity_Class(l_o_s, l_o_s, g, 'gay', 'gayrights_should_be_legalized.')
    o.compute_keyword_sim()
    # c_ob = Keyword_similarity_Class(sentences=l_o_s, graph=None)
    # sent_topic_sim_edges = c_ob.compute_sent_sim()
    # print("sent_topic_sim_edges: ", sent_topic_sim_edges)


