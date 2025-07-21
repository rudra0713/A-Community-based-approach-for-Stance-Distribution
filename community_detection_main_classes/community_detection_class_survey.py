from networkx import ego_graph
import networkx as nx, pickle, sys, os, argparse
sys.path.insert(1, '/scratch/rrs99/Stance_Distribution/')
from berttopic_class import Berttopic_methods
from similarity_class import Similarity_Class
from entailment_class_2 import Entailment_Class
from keyword_extraction_class import Keyword_similarity_Class
from language_model_class import LM_Class
from cdlib import algorithms
from cdlib import viz
import matplotlib.pyplot as plt
from reason.stat_complete import create_article_sentence_map
from networkx.readwrite import json_graph
import json
import numpy as np
# from experiment_codes.gpt3_for_argument_unit_detection_3 import determine_arguments_using_gpt3 as determine_arguments_using_gpt3_3
from experiment_codes.gpt3_for_argument_unit_detection_5 import determine_arguments_using_gpt3 as determine_arguments_using_gpt3_5
import csv
from nltk import word_tokenize
from nltk import sent_tokenize
from cdlib.algorithms import eva
from copy import deepcopy
# from gpt3_for_perspective_detection import generate_perspective_for_arguments
from pruning_ddt import prune_ddt_three_pass_sentence_ddt, prune_ddt_based_on_arg_list
# from svm_class import SVM_class
from datetime import datetime
# from experiment_codes.performance_computation import Performance_analysis

def get_current_time(print_str=''):
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print(print_str + ", Current Time =", current_time)


class Discourse_Graph:
    def __init__(self, read_graph_path='', read_c_a_l_path='', topic_word=None):
        self.graph = nx.Graph()
        self.gold_graph = nx.Graph()
        self.use_topic_modeling = True
        self.claim = None
        self.topic_word = topic_word
        self.pruning_sim = False
        self.pruning_depth = True
        self.pruning_depth_value = 2
        self.pruning_gpt3 = False
        self.articles = []
        self.arg_to_article_map = {}
        self.graph_ob = pickle.load(open(read_graph_path, 'rb'))
        self.c_a_l = pickle.load(open(read_c_a_l_path, 'rb'))
        for id_index in self.c_a_l:
            self.articles.append(self.c_a_l[id_index]['article'])
        self.gpt3_arg_path = 'gpt3_args_version_5'
        if not os.path.exists(self.gpt3_arg_path):
            os.makedirs(self.gpt3_arg_path)

        print("starting main code .. ")
        # sys.exit(0)
        # relabel nodes
        graph_node_counter = 0
        if args.use_only_arguments:
            article_key_to_id_index_mapping = {}
            list_of_articles = [self.c_a_l[id_index]['article_label'] + '.rsn' for id_index in self.graph_ob]
            print("list_of_articles: ", list_of_articles)
            for id_index in self.graph_ob:
                if not self.claim:
                    self.claim = self.graph_ob[id_index]['extra_info']['all_sentences_in_article'][0]

                article_key_to_id_index_mapping[self.c_a_l[id_index]['article_label'] + '.rsn'] = id_index
            # for id_index in self.graph_ob:
            #     article_label = self.c_a_l[id_index]['article_label'].split('.')[0]
            article_sentence_map = create_article_sentence_map(domain='abortion', list_of_articles=list_of_articles)
            print("article sentence map: ", article_sentence_map)
            reason_cluster_id_map = {}
            gold_arguments = []
            arg_reason_to_graph_id_map = {}
            for article_key in article_sentence_map:
                article = self.c_a_l[article_key_to_id_index_mapping[article_key]]['article']
                for arg_reason in article_sentence_map[article_key]:
                    if arg_reason not in reason_cluster_id_map:
                        reason_cluster_id_map[arg_reason] = len(reason_cluster_id_map) + 1
                    for arg in article_sentence_map[article_key][arg_reason]:
                        gold_arguments.append(arg)
                        print("gold argument: ", article_key, '-', arg_reason, '-', arg)
                        self.graph.add_node(graph_node_counter, text=arg, reason_text=arg_reason, stance=self.c_a_l[article_key_to_id_index_mapping[article_key]]['label'])
                        self.arg_to_article_map[int(graph_node_counter)] = {
                            'article_label': article_key,
                            'id_label': article_key_to_id_index_mapping[article_key],
                            'arg_index': str(article.find(arg)) + '-' + str(article.find(arg) + len(arg))
                        }
                        self.gold_graph.add_node(graph_node_counter, text=arg, reason_text=arg_reason, stance=self.c_a_l[article_key_to_id_index_mapping[article_key]]['label'])
                        if arg_reason not in arg_reason_to_graph_id_map:
                            arg_reason_to_graph_id_map[arg_reason] = [graph_node_counter]
                        else:
                            arg_reason_to_graph_id_map[arg_reason].append(graph_node_counter)

                        graph_node_counter += 1

            for arg_reason in arg_reason_to_graph_id_map:
                for i in range(len(arg_reason_to_graph_id_map[arg_reason]) - 1):
                    for j in range(i + 1, len(arg_reason_to_graph_id_map[arg_reason])):
                        self.gold_graph.add_edge(arg_reason_to_graph_id_map[arg_reason][i], arg_reason_to_graph_id_map[arg_reason][j])
            nx.write_gml(self.gold_graph, 'gold_' + args.gold_graph_name + '.gml')
        else:
            for id_index in self.graph_ob:
                print(id_index)
                # stance_label = self.c_a_l[id_index]['label']
                stance_label_ob = {}
                discourse_graph = self.graph_ob[id_index]['sentence_ddt']
                for node in discourse_graph.nodes:
                    stance_label_ob[node] = {'stance': self.c_a_l[id_index]['label']}
                nx.set_node_attributes(discourse_graph, stance_label_ob)
                all_sentences_in_article = self.graph_ob[id_index]['extra_info']['all_sentences_in_article'][1:]
                if not self.claim:
                    self.claim = self.graph_ob[id_index]['extra_info']['all_sentences_in_article'][0]

                if self.pruning_sim:
                    discourse_graph = prune_ddt_three_pass_sentence_ddt(ddt=discourse_graph, graph_path=read_graph_path, claim_edu_list=['1'], id_index=id_index)
                    # sys.exit(0)
                if self.pruning_depth:
                    discourse_graph_root = [node for node in discourse_graph.nodes if discourse_graph.in_degree(node) == 0][0]
                    survived_nodes = ['0', '1']

                    for node in discourse_graph.nodes:
                        if node not in survived_nodes and 'text' in discourse_graph.nodes[node] and len(discourse_graph.nodes[node]['text'].strip().split(" ")) > 2:
                            survived_nodes.append(node)
                    print("survived nodes: ", survived_nodes)
                    discourse_graph = prune_ddt_based_on_arg_list(ddt=discourse_graph, claim_edu_list=['1'],
                                                                  arg_list=survived_nodes)
                    discourse_graph = ego_graph(discourse_graph, n=discourse_graph_root, radius=self.pruning_depth_value)
                    print("discourse graph : ", discourse_graph.nodes, discourse_graph.edges)

                if self.pruning_gpt3:
                    survived_nodes = ['0', '1']
                    survived_args = []
                    try:
                        # this is done to avoid monetary cost
                        # if __xx is inserted, that means we would like to update the previously stored results
                        # version_5 refers to using the fifth version
                        gpt_3_args = pickle.load(open(self.gpt3_arg_path + os.sep + self.claim.lower().replace(' ','_') + '_' + id_index
                                                      + '.p', 'rb'))
                        gpt_3_picked_args = gpt_3_args[id_index]
                        print("successfully loaded args for ", id_index)
                    except:

                        gpt_3_picked_args = determine_arguments_using_gpt3_5(self.c_a_l[id_index]['claim'],
                                                                             all_sentences_in_article)
                        gpt_3_picked_args = [v for v in gpt_3_picked_args if v < len(all_sentences_in_article)]
                        pickle.dump({id_index: gpt_3_picked_args}, open(self.gpt3_arg_path + os.sep + self.claim.lower().replace(' ','_') + '_' + id_index
                                                      + '.p', 'wb'))

                    for pred_ind in gpt_3_picked_args:
                        survived_args.append(all_sentences_in_article[pred_ind])

                    for node in discourse_graph.nodes:
                        if 'text' in discourse_graph.nodes[node] and discourse_graph.nodes[node]['text'] in survived_args:
                            survived_nodes.append(node)
                    print("survived arg: ", survived_args)
                    print("survived nodes: ", survived_nodes)
                    discourse_graph = prune_ddt_based_on_arg_list(ddt=discourse_graph, claim_edu_list=['1'],
                                                                  arg_list=survived_nodes)
                    print("discourse graph : ", discourse_graph.nodes, discourse_graph.edges)
                # print(discourse_graph.nodes(data=True))
                # print(discourse_graph.edges)

                discourse_graph = self.root_claim_removal_from_ddt(discourse_graph)
                # print(discourse_graph.nodes(data=True))
                # print(discourse_graph.edges)
                # sys.exit(0)
                mapping = dict(zip(discourse_graph, range(graph_node_counter, graph_node_counter + len(discourse_graph))))
                # print("mapping: ", mapping)
                discourse_graph = nx.relabel_nodes(discourse_graph, mapping)
                # print(discourse_graph.nodes(data=True))

                # print(discourse_graph.nodes[0]['text'])
                # print("discourse_graph nodes: ", discourse_graph.nodes)
                # print("discourse_graph edges: ", discourse_graph.edges)
                graph_node_counter += len(discourse_graph)
                # print("self.graph type: ", type(self.graph))
                self.graph.add_nodes_from(discourse_graph.nodes(data=True))
                attrs = {}
                for u, v in discourse_graph.edges():
                    attrs[(u, v)] = {"edge_type": 'stance_edge', "info": '0'}
                nx.set_edge_attributes(discourse_graph, attrs)
                self.graph.add_edges_from(discourse_graph.edges(data=True))
                for node in discourse_graph.nodes:
                    # A1.data.rsn -> A1
                    self.arg_to_article_map[int(node)] = {'article_label': self.c_a_l[id_index]['article_label'].split('.')[0],
                                                          'id_label': id_index,
                                                          'arg_index': all_sentences_in_article.index(discourse_graph.nodes[node]['text'])}

                # print(self.graph.nodes(data=True))
                # print("...")
                # sys.exit(0)

        print(self.graph.nodes(data=True))
        print(self.arg_to_article_map)
        print(f"claim: {self.claim}")
        # print(self.graph.nodes[0]['text'])

        self.sentences = [self.graph.nodes[node]['text'] if 'text' in self.graph.nodes[node] else '' for node in self.graph.nodes]
        count_empty = sum([1 for sent in self.sentences if sent == ''])
        assert count_empty == 0
        assert len(self.sentences) == len(self.arg_to_article_map)
        print("count_empty: ", count_empty)
        print("all arguments ..")
        for sent in self.sentences:
            print(sent)
        print("...........")
        # with open('abortion_arguments.txt', 'w') as f:
        #     for line in self.sentences:
        #         f.write(f"{line}\n")
        # tsv_data = [[self.graph.nodes[node]['text'], self.graph.nodes[node]['stance']] for node in self.graph.nodes]
        # with open('abortion_arguments.tsv', 'w') as tsvfile:
        #     # csv writer to write in tsv file
        #     tsv_writer = csv.writer(tsvfile, delimiter='\t')
        #     # write header in tsv file
        #     # write rows
        #     tsv_writer.writerows(tsv_data)
        #     # close csv file
        #     tsvfile.close()
        #     pass
        # print(self.sentences[:5])
        print("arg to article map ... ")
        for node in self.arg_to_article_map:
            print(node, self.arg_to_article_map[node]['article_label'])

        # sys.exit(0)

    def root_claim_removal_from_ddt(self, ddt):
        root_children = list(ddt.successors('0'))
        if len(root_children) == 1:
            ddt.remove_node('0')
        else:
            root_child_index_found = -1
            for root_child_index, root_child in enumerate(root_children):
                dfs_res = list(nx.dfs_edges(ddt, source=root_child))
                for (s, e) in dfs_res:
                    if s == '1' or e == '1':
                        root_child_index_found = root_child_index
                        break
                if root_child_index_found != -1:
                    break
            # print("root child index found -> ", root_child_index_found)
            if root_child_index_found + 1 < len(root_children):
                new_root_id = root_children[root_child_index_found + 1]
            else:
                new_root_id = root_children[root_child_index_found - 1]
            # print("new root id ", new_root_id)
            for root_child in root_children:
                if root_child != new_root_id:
                    ddt.add_edge(new_root_id, root_child)
            ddt.remove_node('0')
        if ddt.has_node('1'):
            claim_node_predecessors = list(ddt.predecessors('1'))
            if len(claim_node_predecessors) == 0:
                if len(list(ddt.successors('1'))) <= 1:
                    pass
                else:
                    claim_node_successor_left = list(ddt.successors('1'))[0]
                    for claim_node_successor in list(ddt.successors('1'))[1:]:
                        ddt.add_edge(claim_node_successor_left, claim_node_successor)
            elif len(claim_node_predecessors) == 1:
                claim_node_predecessor = claim_node_predecessors[0]
                for claim_node_successor in list(ddt.successors('1')):
                    ddt.add_edge(claim_node_predecessor, claim_node_successor)
            ddt.remove_node('1')
        return ddt

    def apply_leiden(self):
        leiden_communities = algorithms.leiden(self.graph)
        pos = nx.spring_layout(self.graph)
        viz.plot_network_clusters(self.graph, leiden_communities, pos, figsize=(5, 5), plot_labels=True)
        plt.savefig('leiden.png')
        for i, com in enumerate(leiden_communities.communities):
            print("leiden community : ", str(i + 1))
            for sent_index in com:
                print(sent_index, self.sentences[sent_index])
            print("...")

    def apply_eva(self, topic_sent_dict):
        cur_graph = deepcopy(self.graph)
        eva_mapping = dict(zip(cur_graph, range(0, len(cur_graph))))
        inv_eva_mapping = {v: k for k, v in eva_mapping.items()}
        cur_graph = nx.relabel_nodes(cur_graph, eva_mapping)
        print("eva mapping: ", eva_mapping)
        for key in eva_mapping:
            self.arg_to_article_map[int(eva_mapping[key])] = self.arg_to_article_map.pop(key)

        print("cur graph nodes: ", len(cur_graph.nodes))

        topic_sent_dict_mapped = {}
        for key in topic_sent_dict:
            topic_sent_dict_mapped[eva_mapping[key]] = topic_sent_dict[key]
        print("topic_sent_dict_mapped keys: ", len(topic_sent_dict_mapped))
        # print("cur graph nodes: ", cur_graph.nodes)
        # print("cur graph edges: ", cur_graph.edges)
        # print("topic_sent_dict_mapped: ", topic_sent_dict_mapped)
        # pickle.dump({'graph': cur_graph, 'label': topic_sent_dict_mapped}, open('eva_trial.p', 'wb'))
        eva_communities = eva(cur_graph, topic_sent_dict_mapped, alpha=0.5)
        get_current_time('Finished applying eva algorithm .. ')

        pos = nx.spring_layout(cur_graph)
        viz.plot_network_clusters(cur_graph, eva_communities, pos, figsize=(5, 5), plot_labels=True)
        plt.savefig('eva.png')
        print("eva_communities: ", eva_communities.communities)

        perspective_outputs = []
        for i, com in enumerate(eva_communities.communities):
            com = [inv_eva_mapping[v] for v in com]
            # print("com: ", com)
            # func_out = self.process_output(com)  # is a list, every element is an index followed by argument
            func_out = self.process_output_no_merging(com)  # is a list, every element is an index followed by argument

            perspective_outputs.append(func_out)  # is a list of list, every element is for one community

        get_current_time('Finished processing outputs .. ')

        # perspective_labels = generate_perspective_for_arguments(perspective_outputs)
        perspective_labels = ['empty label'] * len(perspective_outputs)
        # assert len(perspective_labels) == len(perspective_outputs)
        get_current_time('Finished gpt3 perspective .. ')

        # if len(perspective_labels) != len(perspective_outputs):
        #     print("PERSPECTIVE LABELS LENGTH DO NOT MATCH THE LENGTH OF PERSPECTIVE OUTPUTS .. ")
        #     if len(perspective_labels) < len(perspective_outputs):
        #         perspective_labels.extend(['Empty Label'] * (len(perspective_outputs) - len(perspective_labels)))
        #     else:
        #         perspective_labels = perspective_labels[:len(perspective_outputs)]
        gen_data = {
            "eva_communities": eva_communities.communities,
            "topic": self.topic_word
        }
        predicted_graph = nx.read_gml(predicted_graph_name + '.gml')
        gen_data["predicted_graph"] = json_graph.node_link_data(predicted_graph)  # directly storing the graph causing weird error
        gen_data["community_args"] = (perspective_outputs, perspective_labels)
        with open(output_file_name + ".json", "w") as file_o:
            json.dump(gen_data, file_o)


        # pickle.dump((perspective_outputs, perspective_labels), open(args.output_file_name + '.p', 'wb'))
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        svm_dir_path = 'svm-outputs-' + self.claim + current_time

        for i, (pers_out, label) in enumerate(zip(perspective_outputs, perspective_labels)):
            print("eva community : ", str(i + 1))
            print("perspective: ", label)
            for ind, arg, stance_label in pers_out:
                print(ind, arg, stance_label)
            # print("after svm ")
            # svm_ob = SVM_class(claim=self.claim, sentences=pers_out, community_index=i+1, svm_dir_path=svm_dir_path)
            # svm_ob.build_svm_figures()
            # get_current_time('Finished svm for community ' + str(i + 1) + ' ..')

            print("...")

            # com_args = [self.sentences[sent_index] for sent_index in com]
            # # print("perspective: ", generate_perspective_for_arguments(com_args))
            # pro_count = 0
            # for sent_index in com:
            #     # print(sent_index, self.sentences[sent_index])
            #     print(self.arg_to_article_map[sent_index]['article_label'], '-', str(self.arg_to_article_map[sent_index]['arg_index']), self.sentences[sent_index])
            #     if self.c_a_l[self.arg_to_article_map[sent_index]['id_label']]['label'] == 0:
            #         pro_count += 1
            # print("pro distribution: ", round(pro_count / len(com), 2))

    def process_output(self, community):
        arg_info = [(self.arg_to_article_map[sent_index]['article_label'], self.arg_to_article_map[sent_index]['arg_index']) for sent_index in community]
        article_to_arg_map = {(self.arg_to_article_map[sent_index]['article_label'], self.arg_to_article_map[sent_index]['arg_index']): sent_index for sent_index in community}
        arg_info = sorted(arg_info, key=lambda element: (element[0], element[1]))
        com_args = []
        com_args_article = []
        start_index, end_index = -1, -1
        current_article_id = ''
        current_arg = ''
        for i in range(len(arg_info)):
            if i == 0:
                current_arg += self.sentences[article_to_arg_map[arg_info[i]]]
                current_article_id = arg_info[i][0]
                start_index = arg_info[i][1]
                end_index = arg_info[i][1]
            elif arg_info[i][0] != arg_info[i-1][0] or arg_info[i][1] - arg_info[i-1][1] > 1:  # from different article
                com_args.append(current_arg)
                if start_index == end_index:
                    com_args_article.append(current_article_id + ', ' + str(start_index))
                else:
                    com_args_article.append(current_article_id + ', ' + str(start_index) + '-' + str(end_index))
                current_arg = self.sentences[article_to_arg_map[arg_info[i]]]
                current_article_id = arg_info[i][0]
                start_index = arg_info[i][1]
                end_index = arg_info[i][1]
            else:
                end_index += 1
                current_arg += ' ' + self.sentences[article_to_arg_map[arg_info[i]]]
        if len(current_arg) > 0:
            com_args.append(current_arg)
            if start_index == end_index:
                com_args_article.append(current_article_id + ', ' + str(start_index))
            else:
                com_args_article.append(current_article_id + ', ' + str(start_index) + '-' + str(end_index))
        assert len(com_args) == len(com_args_article)
        output_ind_arg_stance = []
        for ind, arg in zip(com_args_article, com_args):
            art_label = ind.split(",")[0]
            # print("art_label: ", art_label)
            for k in self.arg_to_article_map:
                # print(self.arg_to_article_map[k]['article_label'])
                if self.arg_to_article_map[k]['article_label'] == art_label:
                    stance_label = self.c_a_l[self.arg_to_article_map[k]['id_label']]['label']
                    output_ind_arg_stance.append((ind, arg, stance_label))
                    break
        return output_ind_arg_stance

    def process_output_no_merging(self, community):
        output_ind_arg_stance = []
        for sent_index in community:
            ind = str(self.arg_to_article_map[sent_index]['article_label']) + str(self.arg_to_article_map[sent_index]['arg_index'])
            stance_label = self.c_a_l[self.arg_to_article_map[sent_index]['id_label']]['label']
            arg = self.graph.nodes[sent_index]['text']
            output_ind_arg_stance.append((ind, arg, stance_label))

        # arg_info = [(self.arg_to_article_map[sent_index]['article_label'], self.arg_to_article_map[sent_index]['arg_index']) for sent_index in community]
        #
        # article_to_arg_map = {(self.arg_to_article_map[sent_index]['article_label'], self.arg_to_article_map[sent_index]['arg_index']): sent_index for sent_index in community}
        # output_ind_arg_stance = []
        #
        # for a_info in arg_info:
        #     ind = str(a_info[0]) + ',' + str(a_info[1])
        #     stance_label = self.c_a_l[self.arg_to_article_map[article_to_arg_map[a_info]]['id_label']]['label']
        #     arg = self.sentences[article_to_arg_map[a_info]]
        #     output_ind_arg_stance.append((ind, arg, stance_label))

        return output_ind_arg_stance

    def run(self):
        print("current edges: ", self.graph.edges)
        print("current node count: ", len(self.graph.nodes))
        if not args.use_stance_tree:
            print("removing stance tree edges: ")
            self.graph.remove_edges_from(self.graph.edges())
            print("current edges: ", self.graph.edges)

        if args.use_topic_similarity:
            get_current_time('Invoking Berttopic methods .. ')
            bt_topic = Berttopic_methods()
            bt_topic.compute_topic_embeddings(self.claim, self.topic_word)
            # bt_topic.train_berttopic_with_wiki_articles(self.claim)
            topic_sent_dict, topic_edges = bt_topic.test_berttopic_to_add_edges_updated(self.sentences)
            print("topic edges: ", topic_edges)
            print("total topic edges: ", len(topic_edges))
            for node_u, node_v, prop in topic_edges:
                self.graph.add_edge(node_u, node_v, edge_type='topic_similarity', info=prop)

            # self.graph.add_edges_from(topic_edges, edge_type='topic_similarity')
            print("all edges after adding topic edges: ", self.graph.edges)
            get_current_time('Finished adding topic edges .. ')
        if args.use_semantic_similarity:
            sim_class_ob = Similarity_Class(sentences=self.sentences, graph=self.graph)
            sent_topic_sim_edges = sim_class_ob.compute_sent_sim()
            print("sent_topic_sim_edges: ", sent_topic_sim_edges)
            print("total semantic similar edges: ", len(sent_topic_sim_edges))
            for node_u, node_v, prop in sent_topic_sim_edges:
                self.graph.add_edge(node_u, node_v, edge_type='semantic_similarity', info=prop)

            # self.graph.add_edges_from(sent_topic_sim_edges, edge_type='semantic_similarity')
            print("all edges after adding similarity edges: ", self.graph.edges)
            get_current_time('Finished adding similarity edges .. ')

        if args.use_entailment:
            entailment_class_ob = Entailment_Class(sentences=self.sentences, graph=self.graph)
            entailment_edges = entailment_class_ob.run_model()
            print("entailment_edges: ", entailment_edges)
            self.graph.add_edges_from(entailment_edges, edge_type='entailment')
            print("total entailment edges: ", len(entailment_edges))
            print("all edges after adding entailment edges: ", self.graph.edges)
            get_current_time('Finished adding entailment edges .. ')
        if args.use_keyword_similarity:
            keyword_class_ob = Keyword_similarity_Class(sentences=self.sentences, articles=self.articles, graph=self.graph, topic_word=self.topic_word, claim=self.claim)
            keyword_edges = keyword_class_ob.compute_keyword_sim()
            entity_edges = keyword_class_ob.add_entity_edges()
            print("keyword_edges: ", keyword_edges)
            print("total keyword edges: ", len(keyword_edges))
            print("entity_edges: ", entity_edges)
            print("total entity edges, may have overlap with keyword edges: ", len(entity_edges))
            # self.graph.add_edges_from(keyword_edges, edge_type='keyword_similarity')
            for node_u, node_v, prop in entity_edges:
                self.graph.add_edge(node_u, node_v, edge_type='entity_similarity', info=prop)

            for node_u, node_v, prop in keyword_edges:
                if not self.graph.has_edge(node_u, node_v):
                    self.graph.add_edge(node_u, node_v, edge_type='keyword_similarity', info=prop)

            print("all edges after adding keyword edges: ", self.graph.edges)
            get_current_time('Finished adding keyword edges .. ')
        if args.use_lm_score:
            lm_class_ob = LM_Class(sentences=self.sentences, graph=self.graph)
            lm_edges = lm_class_ob.compute_lm_edges()
            print("lm_edges: ", lm_edges)
            print("total lm edges: ", len(lm_edges))
            self.graph.add_edges_from(lm_edges, edge_type='lm')



        # topic_jac_edges = sim_class_ob.compute_jaccard()
        # print("topic jac edges: ", topic_jac_edges)
        #
        # self.graph.add_edges_from(topic_jac_edges)
        nx.write_gml(self.graph, predicted_graph_name + '.gml')
        print("total graph edges: ", self.graph.number_of_edges())
        # self.apply_leiden()
        # print("edge count: ", len(topic_edges), len(sent_topic_sim_edges), len(entailment_edges), self.graph.number_of_edges())
        # print("percentage of topic similarity edges: ", len(topic_edges) / self.graph.number_of_edges())
        # print("percentage of semantic similarity edges: ", len(sent_topic_sim_edges) / self.graph.number_of_edges())
        # print("percentage of entailment edges: ", len(entailment_edges) / self.graph.number_of_edges())

        # print("edge not added ...... ")
        # for i in range(len(self.sentences) - 1):
        #     for j in range(i + 1, len(self.sentences)):
        #         if self.graph.has_edge(i, j) or self.graph.has_edge(j, i) or \
        #                 self.arg_to_article_map[i]['article_label'] == self.arg_to_article_map[j]['article_label']:
        #             pass
        #         else:
        #             pass
        #             print("no edge between: ", i, self.sentences[i], '--', j, self.sentences[j], '--')
        print("is graph directed: ", nx.is_directed(self.graph))
        # p_a = Performance_analysis(read_graph_path=args.read_graph_path, read_c_a_l_path=args.read_c_a_l_path, result_path=args.output_file_name, gold_graph_path='gold_' + args.gold_graph_name + '.gml', predicted_graph_path='predicted_' + args.predicted_graph_name + '.gml')
        if args.use_topic_similarity:
            # pickle.dump(topic_sent_dict, open('topic_sent_dict.p', 'wb'))
            self.apply_eva(topic_sent_dict=topic_sent_dict)


# dp = Discourse_Graph(read_graph_path='/scratch/rrs99/Discourse_parser_pipeline/output_data/discourse_full_procon_all.xlsx+social_networking.p')
# dp = Discourse_Graph(read_graph_path='/scratch/rrs99/Discourse_parser_pipeline/output_data_procon_all.xlsx+gun_control/discourse_full_procon_all.xlsx+gun_control.p')
# dp = Discourse_Graph(read_graph_path='/scratch/rrs99/Discourse_parser_pipeline/output_data_reason+abortion/discourse_full_reason+abortion.p',
#                      read_c_a_l_path='/scratch/rrs99/Discourse_parser_pipeline/output_data_reason+abortion/claim_article_label.p')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Community Detection pipeline')
    parser.add_argument('--read_graph_path', type=str, default="")
    parser.add_argument('--read_c_a_l_path', type=str, default="")
    parser.add_argument('--topic_word', type=str, default="")
    parser.add_argument('--output_file_name', type=str)
    parser.add_argument('--gold_graph_name', type=str)
    parser.add_argument('--predicted_graph_name', type=str)
    parser.add_argument('--use_topic_similarity', type=str, default='True')
    parser.add_argument('--use_semantic_similarity', type=str, default='True')
    parser.add_argument('--use_entailment', type=str, default='True')
    parser.add_argument('--use_keyword_similarity', type=str, default='True')
    parser.add_argument('--use_lm_score', type=str, default='True')
    parser.add_argument('--use_stance_tree', type=str, default='True')
    parser.add_argument('--use_only_arguments', type=str, default='False')
    parser.add_argument('--domain_index', type=str, default="1")

    args = parser.parse_args()

    # Manually convert the string to a boolean value
    args.use_topic_similarity = args.use_topic_similarity.lower() == 'true'
    args.use_semantic_similarity = args.use_semantic_similarity.lower() == 'true'
    args.use_entailment = args.use_entailment.lower() == 'true'
    args.use_stance_tree = args.use_stance_tree.lower() == 'true'
    args.use_only_arguments = args.use_only_arguments.lower() == 'true'
    args.use_keyword_similarity = args.use_keyword_similarity.lower() == 'true'
    args.use_lm_score = args.use_lm_score.lower() == 'true'

    topic_list = ['politics', 'elections', 'gun_control_and_gun_rights', 'immigration', 'us_congress',
                  'foreign_policy', 'healthcare', 'environment', 'terrorism', 'education', 'supreme_court', 'national_security']
    domain_index = int(args.domain_index) - 1
    # if domain_index == 5:
    args.read_graph_path = args.read_graph_path + topic_list[domain_index] + '_200_articles/discourse_full_allsides+' + topic_list[domain_index] + '_200_articles.p'
    args.read_c_a_l_path = args.read_c_a_l_path + topic_list[domain_index] + '_200_articles/claim_article_label.p'
    args.topic_word = topic_list[domain_index]
    output_file_name = 'output_survey_' + topic_list[domain_index] + '_' + args.output_file_name
    predicted_graph_name = 'predicted_survey_' + topic_list[domain_index] + '_' + args.predicted_graph_name
    # dp = Discourse_Graph(read_graph_path='/scratch/rrs99/Discourse_parser_pipeline/output_data_reason+abortion_50_articles/discourse_full_reason+abortion_50_articles.p',
    #                      read_c_a_l_path='/scratch/rrs99/Discourse_parser_pipeline/output_data_reason+abortion_50_articles/claim_article_label.p',
    #                      topic_word='abortion')
    print("read all parameters")
    dp = Discourse_Graph(read_graph_path=args.read_graph_path, read_c_a_l_path=args.read_c_a_l_path,
                         topic_word=args.topic_word)
    # dp = Discourse_Graph(read_graph_path=read_graph_path, read_c_a_l_path=read_c_a_l_path, topic_word=topic_word)

    dp.run()





