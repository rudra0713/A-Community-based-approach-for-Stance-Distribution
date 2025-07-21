import networkx as nx
import copy, pickle
from community_detection_main_classes.compute_claim_sentence_similarity import compute_claim_sentence_sim


similarity_threshold = 0.4


def prune_ddt_similarity_pass(ddt, claim_sentence_list, id_index, graph_path):
    sim_imp_nodes = []
    sim_scores = compute_claim_sentence_sim(id_index, graph_path, None)

    bfs_res = list(nx.bfs_edges(ddt, source='0'))
    initial_bfs_res = copy.deepcopy(bfs_res)
    for (s, e) in initial_bfs_res:
        if sim_scores[int(e)] > similarity_threshold and e not in claim_sentence_list:
            sim_imp_nodes.append(e)
    # print("similarity important nodes ", sim_imp_nodes)

    return sim_imp_nodes


def prune_ddt_three_pass_sentence_ddt(ddt, graph_path, claim_edu_list, id_index):
    # global summary_side_evaluation, nodes_before_sim_pruning, nodes_after_sim_pruning, should_do_pruning, ddt_for_summary, summary_sentences_mapping, org_to_new_mapping, print_specific

    ##### just to check something starts
    root_node = [n for n, d in ddt.in_degree() if d == 0][0]
    preorder_nodes = [node for node in list(nx.dfs_preorder_nodes(ddt, source=root_node))]
    # print("preorder nodes after starting  three pass -> ", preorder_nodes)
    ##### just to check something ends
    number_of_nodes_original = len(ddt.nodes)
    # print("number_of_nodes_original , nodes_before_sim_pruning ", number_of_nodes_original, nodes_before_sim_pruning)
    # print("ddt nodes ", ddt.nodes)
    sim_nodes = prune_ddt_similarity_pass(ddt, claim_edu_list, id_index, graph_path)

    imp_nodes = list(set(sim_nodes) | set(claim_edu_list) | {'0'})

    bfs_res = list(nx.bfs_edges(ddt, source='0'))
    initial_bfs_res = copy.deepcopy(bfs_res)
    # # print("bfs res ", bfs_res)
    layers = [['0']]
    current_root = ['0']
    while len(current_root) > 0:
        new_layer = []
        new_roots = []
        for root in current_root:
            for (src, dest) in bfs_res:
                if src == root:
                    new_layer.append(dest)
                    new_roots.append(dest)
        if len(new_layer) > 0:
            layers.append(new_layer)
        current_root = copy.deepcopy(new_roots)
    # print("layers ", layers)
    layers.reverse()
    # sys.exit(0)
    special_nodes = []
    for layer in layers:
        for parent in layer:
            # # print("parent ", parent)
            # if parent in claim_edu_list:
            #     # print("skipping ", parent, "since it's part of the claim")
            #     continue
            children = list(ddt.successors(parent))
            if len(children) == 0:
                continue
            if len(children) != 0:
                if parent not in imp_nodes:
                    survived_nodes = []
                    for dest in children:
                        if dest not in imp_nodes and dest not in special_nodes:
                            ddt.remove_edge(parent, dest)
                            ddt.remove_node(dest)
                            # # print("removing edge between", parent, ",", dest)
                        else:
                            survived_nodes.append(dest)
                    if len(survived_nodes) == 0:
                        ddt.remove_node(parent)  # thus removing the entire subtree
                    else:
                        # parent is nucleus, and at least one of the survived children is satellite
                        # then, do not delete the nucleus
                        at_least_one_child_satellite = False
                        for survived_node in survived_nodes:
                            if ddt.nodes[survived_node]['nuclearity'] == 'S':
                                at_least_one_child_satellite = True
                                break
                        if ddt.nodes[parent]['nuclearity'] == 'N' and at_least_one_child_satellite:
                            special_nodes.append(parent)  # make sure, this node does not get deleted later
                            pass
                        else:
                            # bring the left most nucleus child to the beginning of the survived nodes
                            for survived_node in survived_nodes:
                                if ddt.nodes[survived_node]['nuclearity'] == 'N':
                                    survived_nodes.remove(survived_node)
                                    survived_nodes.insert(0, survived_node)
                                    break
                            # create connection from the leftmost child (new parent) to it's siblings

                            for survived_node in survived_nodes[1:]:
                                ddt.add_edge(survived_nodes[0], survived_node)
                            # # print("adding edge from parent's parent -> ", list(ddt.predecessors(parent))[0])
                            ddt.add_edge(list(ddt.predecessors(parent))[0], survived_nodes[0])
                            ddt.remove_node(parent)

                else:
                    for dest in children:
                        if dest not in imp_nodes and dest not in special_nodes:
                            ddt.remove_edge(parent, dest)
                            ddt.remove_node(dest)
                            # # print("removing edge between", parent, ",", dest)
                # # print("nodes ", ddt.nodes)
    # print("special nodes ", special_nodes)
        imp_nodes = imp_nodes + special_nodes
    #
    # # print("inside ddt nodes", ddt.nodes)
    # # print("sim nodes ", sim_nodes)
    number_of_nodes_after_pruning = len(ddt.nodes)
    print(" nodes before and after pruning ", number_of_nodes_original, number_of_nodes_after_pruning)
    # if print_specific:
    #     drawing_ddt_3pass(ddt, id_index, store_image_directory, '_summary_org', None, True)

    return ddt


def prune_ddt_based_on_arg_list(ddt, claim_edu_list, arg_list=[]):
    # global summary_side_evaluation, nodes_before_sim_pruning, nodes_after_sim_pruning, should_do_pruning, ddt_for_summary, summary_sentences_mapping, org_to_new_mapping, print_specific

    ##### just to check something starts
    root_node = [n for n, d in ddt.in_degree() if d == 0][0]
    preorder_nodes = [node for node in list(nx.dfs_preorder_nodes(ddt, source=root_node))]
    # print("preorder nodes after starting  three pass -> ", preorder_nodes)
    ##### just to check something ends
    number_of_nodes_original = len(ddt.nodes)
    # print("number_of_nodes_original , nodes_before_sim_pruning ", number_of_nodes_original, nodes_before_sim_pruning)
    # print("ddt nodes ", ddt.nodes)
    # sim_nodes = prune_ddt_similarity_pass(ddt, claim_edu_list, id_index, graph_path)

    imp_nodes = list(set(arg_list) | set(claim_edu_list) | {'0'})

    bfs_res = list(nx.bfs_edges(ddt, source='0'))
    initial_bfs_res = copy.deepcopy(bfs_res)
    # # print("bfs res ", bfs_res)
    layers = [['0']]
    current_root = ['0']
    while len(current_root) > 0:
        new_layer = []
        new_roots = []
        for root in current_root:
            for (src, dest) in bfs_res:
                if src == root:
                    new_layer.append(dest)
                    new_roots.append(dest)
        if len(new_layer) > 0:
            layers.append(new_layer)
        current_root = copy.deepcopy(new_roots)
    # print("layers ", layers)
    layers.reverse()
    # sys.exit(0)
    special_nodes = []
    for layer in layers:
        for parent in layer:
            # # print("parent ", parent)
            # if parent in claim_edu_list:
            #     # print("skipping ", parent, "since it's part of the claim")
            #     continue
            children = list(ddt.successors(parent))
            if len(children) == 0:
                continue
            if len(children) != 0:
                if parent not in imp_nodes:
                    survived_nodes = []
                    for dest in children:
                        if dest not in imp_nodes and dest not in special_nodes:
                            ddt.remove_edge(parent, dest)
                            ddt.remove_node(dest)
                            # # print("removing edge between", parent, ",", dest)
                        else:
                            survived_nodes.append(dest)
                    if len(survived_nodes) == 0:
                        ddt.remove_node(parent)  # thus removing the entire subtree
                    else:
                        # # parent is nucleus, and at least one of the survived children is satellite
                        # # then, do not delete the nucleus
                        # at_least_one_child_satellite = False
                        # for survived_node in survived_nodes:
                        #     if ddt.nodes[survived_node]['nuclearity'] == 'S':
                        #         at_least_one_child_satellite = True
                        #         break
                        # if ddt.nodes[parent]['nuclearity'] == 'N' and at_least_one_child_satellite:
                        #     special_nodes.append(parent)  # make sure, this node does not get deleted later
                        #     pass
                        # else:
                        #     # bring the left most nucleus child to the beginning of the survived nodes
                        for survived_node in survived_nodes:
                            if ddt.nodes[survived_node]['nuclearity'] == 'N':
                                survived_nodes.remove(survived_node)
                                survived_nodes.insert(0, survived_node)
                                break
                        # create connection from the leftmost child (new parent) to it's siblings

                        for survived_node in survived_nodes[1:]:
                            ddt.add_edge(survived_nodes[0], survived_node)
                        # # print("adding edge from parent's parent -> ", list(ddt.predecessors(parent))[0])
                        ddt.add_edge(list(ddt.predecessors(parent))[0], survived_nodes[0])
                        ddt.remove_node(parent)

                else:
                    for dest in children:
                        if dest not in imp_nodes and dest not in special_nodes:
                            ddt.remove_edge(parent, dest)
                            ddt.remove_node(dest)
                            # # print("removing edge between", parent, ",", dest)
                # # print("nodes ", ddt.nodes)
    # print("special nodes ", special_nodes)
        imp_nodes = imp_nodes + special_nodes
    #
    # # print("inside ddt nodes", ddt.nodes)
    # # print("sim nodes ", sim_nodes)
    number_of_nodes_after_pruning = len(ddt.nodes)
    print(" nodes before and after pruning ", number_of_nodes_original, number_of_nodes_after_pruning)
    # if print_specific:
    #     drawing_ddt_3pass(ddt, id_index, store_image_directory, '_summary_org', None, True)

    return ddt
