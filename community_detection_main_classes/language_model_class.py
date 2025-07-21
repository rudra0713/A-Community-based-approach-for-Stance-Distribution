import networkx as nx
import torch, random
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class LM_Class:
    def __init__(self, sentences, graph):
        self.sentences = sentences
        self.graph = graph
        self.lm_score_threshold = 4  # determined after looking 200 articles on abortion

    def calculate_likelihood(self, sequence):
        # Load pre-trained GPT-2 model and tokenizer
        model_name = 'gpt2'
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)

        # Tokenize the sequence and convert to tensor
        input_ids = tokenizer.encode(sequence, return_tensors='pt')

        # Get the log likelihood of the sequence from the model
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            log_likelihood = outputs.loss.item()

        return log_likelihood

    def compute_lm_edges(self):
        lm_edges = []
        for i in range(len(self.sentences) - 1):
            for j in range(i + 1, len(self.sentences)):
                sequence = self.sentences[i] + ' ' + self.sentences[j]
                if not self.graph.has_edge(i, j) and self.calculate_likelihood(sequence) >= self.lm_score_threshold:
                    lm_edges.append((i, j))
        return lm_edges
