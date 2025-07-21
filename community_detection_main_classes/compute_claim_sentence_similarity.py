import pickle
from similarity_score_codes.return_bert_sim_score_2 import compute_sim_score


def compute_claim_sentence_sim(key, read_graph_path, failed_ids_path=None):
    graph_ob = pickle.load(open(read_graph_path, 'rb'))
    all_failed_ids = []
    if failed_ids_path:
        all_failed_ids = pickle.load(open(failed_ids_path, "rb"))
        all_failed_ids = [id_index for (id_index, _) in all_failed_ids]

    print("len of claim article label ", len(graph_ob))
    if key not in all_failed_ids:
        article_info = graph_ob[key]['extra_info']['all_sentences_in_article']

        claim = article_info[0]
        article_sentences = article_info
        print("claim: ", claim)
        for sent in article_sentences[1:]:
            print(sent)
        sentence_sim = compute_sim_score([claim], article_sentences, compute_max=False)
        # adding one zero to set claim-root similarity to zero
        sentence_sim.insert(0, 0)
        print("sentence sim: ", sentence_sim)
        return sentence_sim
    return []

