from transformers import pipeline, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
import torch
import networkx as nx

class Entailment_Class:
    def __init__(self, sentences, graph):
        self.sentences = sentences
        self.entailment_threshold = 0.90
        self.graph = graph
        self.entailment_data = []
        self.prepare_data()

    def prepare_data(self):
        for i in range(len(self.sentences)):
            for j in range(len(self.sentences)):
                self.entailment_data.append((i, j, self.sentences[i] + ' ' + self.sentences[j]))

    def run_model(self):
        device = 0 if torch.cuda.is_available() else -1

        # model = AutoModelForSequenceClassification.from_pretrained('/scratch/rrs99/Stance_Distribution/roberta-large-mnli')
        # tokenizer = AutoTokenizer.from_pretrained('/scratch/rrs99/Stance_Distribution/roberta-large-mnli')

        model = AutoModelForSequenceClassification.from_pretrained('/scratch/rrs99/Stance_Distribution/bart-fined-tuned-on-entailment-classification')
        tokenizer = AutoTokenizer.from_pretrained('/scratch/rrs99/Stance_Distribution/bart-fined-tuned-on-entailment-classification')

        pipe = pipeline(task='text-classification', tokenizer=tokenizer, model=model, device=device)
        candidate_labels = ['NEUTRAL', 'ENTAILMENT', 'CONTRADICTION']
        # pipe = pipeline(model="roberta-large-mnli", device=device) # works if I run with sh

        output = pipe([sent_pair for _, _, sent_pair in self.entailment_data])
        # output = pipe(self.entailment_data[0][2], candidate_labels)
        # print("output", output)
        entailment_edge_between_sentences = []
        for data, out in zip(self.entailment_data, output):
            index_i, index_j, _ = data
            if index_i != index_j and (index_j, index_i) not in entailment_edge_between_sentences and not self.graph.has_edge(index_i, index_j) and\
                    not self.graph.has_edge(index_j, index_i):
                if (out['label'] in ['ENTAILMENT', 'CONTRADICTION'] or out['label'] in ['entailment', 'contradiction']) and out['score'] > self.entailment_threshold:
                    entailment_edge_between_sentences.append((index_i, index_j))
                    print("adding entailment edge between ", index_i, self.sentences[index_i], '--', index_j, self.sentences[index_j], out)

        return entailment_edge_between_sentences
#
# classifier = pipeline('zero-shot-classification', model='roberta-large-mnli')
#
# # You can then use this pipeline to classify sequences into any of the class names you specify. For example:
#
# sequence_to_classify = "one day I will see the world"
# candidate_labels = ['travel', 'cooking', 'dancing']
# print(classifier(sequence_to_classify, candidate_labels))
#
# sequence_to_classify = "I like you. I love you."
# candidate_labels = ['NEUTRAL', 'ENTAILMENT', 'CONTRADICTION']
# print(classifier(sequence_to_classify, candidate_labels))
#
# print(pipe("This restaurant is awesome"))
# print(pipe("I like you. I love you."))
# print(pipe(["Abortion should be legalized. I support abortion.", "Abortion should be legalized. I do not support abortion."]))
# print(pipe())


if __name__ == '__main__':
    pass
    # x = Entailment_Class(sentences=['Abortion should be legalized.', 'I support abortion.'], graph=nx.Graph())
    # x.run_model()
