# import spacy
# import scispacy
# from scispacy.abbreviation import AbbreviationDetector
#
# # Load the spaCy NER model
# nlp = spacy.load("en_core_web_sm")  # Or you can use a larger model like "en_core_web_trf"
#
# # Add SciSpacy's abbreviation resolver to the pipeline
# # You need a scientific or biomedical model from SciSpacy for this step
# sci_nlp = spacy.load("en_core_sci_sm")  # SciSpacy model for abbreviation resolution
# sci_nlp.add_pipe("abbreviation_detector")  # Add the abbreviation detector component
#
# # Example text
text = """
The NRA is a major organization in the United States. It has been involved in various campaigns. Donald Trump and Joe Biden have both mentioned the NRA in their speeches.
Trump also mentioned that he is going to win the election. 
"""
text_list = ["The NRA is a major organization in the United States.", "It has been involved in various campaigns.", "Donald Trump and Joe Biden have both mentioned the NRA in their speeches.", "Trump also mentioned that he is going to win the election."]

#
# # Step 1: Named Entity Recognition with spaCy
# doc = nlp(text)
#
# print("Named Entities recognized by spaCy:")
# for ent in doc.ents:
#     print(ent.text, ent.label_)  # Display the entity and its type (e.g., PERSON, ORG)
#
# # Step 2: Abbreviation Detection and Resolution with SciSpacy
# sci_doc = sci_nlp(text)
#
# print("\nAbbreviations detected by SciSpacy:")
# for abrv in sci_doc._.abbreviations:
#     print(f"{abrv} -> {abrv._.long_form}")
# #
# #
# # # import spacy
# #
# # from scispacy.abbreviation import AbbreviationDetector
# # #
# # # nlp = spacy.load("en_core_sci_sm")
# # #
# # # # Add the abbreviation pipe to the spacy pipeline.
# # # nlp.add_pipe("abbreviation_detector")
# # #
# # # # works for the following but not for the paragraph above
# # # doc = nlp("Spinal and bulbar muscular atrophy (SBMA) is an \
# # #            inherited motor neuron disease caused by the expansion \
# # #            of a polyglutamine tract within the androgen receptor (AR). \
# # #            SBMA can be caused by this easily.")
# # #
# # # print("Abbreviation", "\t", "Definition")
# # # for abrv in doc._.abbreviations:
# # # 	print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")
# #
# # from transformers import pipeline
# #
# # # Load entity linking pipeline
# # entity_linker = pipeline(task="entity-linking", model="hf-internal-testing/tiny-random-wikineural")
# #
# # text = """
# # The NRA, which stands for the National Rifle Association, is a major organization in the United States.
# # It has been involved in various campaigns. Donald Trump and Joe Biden have both mentioned the NRA in their speeches.
# # """
# #
# # # Perform entity linking
# # linked_entities = entity_linker(text)
# # print("linked entities using transformers")
# # print(linked_entities)
#
# import spacy
# # # import neuralcoref
# from transformers import pipeline
# #
# # # # Load spaCy model
# # # nlp = spacy.load("en_core_web_sm")
# # #
# # # # Add NeuralCoref to spaCy pipeline
# # # neuralcoref.add_to_pipe(nlp)
# #
# # # Load Hugging Face NER pipeline
# ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
# #
# #
# # def perform_entity_linking(text):
# #     # Process the text with spaCy and NeuralCoref
# #     # doc = nlp(text)
# #     #
# #     # # Get coreferences
# #     # if doc._.has_coref:
# #     #     print("Coreferences:")
# #     #     for cluster in doc._.coref_clusters:
# #     #         print(f"{cluster.main.text}: {[mention.text for mention in cluster.mentions]}")
# #
# #     # Perform Named Entity Recognition with Hugging Face
# #     ner_results = ner_pipeline(text)
# #
# #     # Group entities
# #     entities = {}
# #     for result in ner_results:
# #         entity = result['word']
# #         entity_type = result['entity']
# #         if entity_type not in entities:
# #             entities[entity_type] = set()
# #         entities[entity_type].add(entity)
# #
# #     # Print entities
# #     print("\nNamed Entities:")
# #     for entity_type, entity_set in entities.items():
# #         print(f"{entity_type}: {', '.join(entity_set)}")
# #
# #     # Simple abbreviation linking
# #     words = text.split()
# #     for i, word in enumerate(words):
# #         if word.isupper() and len(word) > 1:
# #             # Check if the next words form the full name
# #             for j in range(i + 1, len(words)):
# #                 if all(word.startswith(w[0].upper()) for w in words[i + 1:j + 1]):
# #                     print(f"\nPossible abbreviation link: {word} - {' '.join(words[i + 1:j + 1])}")
# #                     break
# #
# #
# # # Example usage
# # text = "The NRA, which stands for the National Rifle Association, is a major organization in the United States. It has been involved in various campaigns. Donald Trump and Joe Biden have both mentioned the NRA in their speeches."
# #
# # perform_entity_linking(text)
# # def perform_entity_linking(text):
# #     # Process the text with spaCy and NeuralCoref
# #     doc = nlp(text)
# #
# #     # Get coreferences
# #     coreferences = {}
# #     if doc._.has_coref:
# #         for cluster in doc._.coref_clusters:
# #             main_entity = cluster.main.text
# #             coreferences[main_entity] = [mention.text for mention in cluster.mentions]
# #
# #     # Perform Named Entity Recognition with Hugging Face
# #     ner_results = ner_pipeline(text)
# #
# #     # Group entities, incorporating coreference information
# #     entities = {}
# #     for result in ner_results:
# #         entity = result['word']
# #         entity_type = result['entity']
# #         if entity_type not in entities:
# #             entities[entity_type] = set()
# #         entities[entity_type].add(entity)
# #
# #         # Add coreferent mentions to the entity set
# #         if entity in coreferences:
# #             entities[entity_type].update(coreferences[entity])
# #
# #     # Print entities with their coreferences
# #     print("\nNamed Entities (including coreferences):")
# #     for entity_type, entity_set in entities.items():
# #         print(f"{entity_type}: {', '.join(entity_set)}")
#
# import spacy
#
#
# def perform_entity_linking(text):
#     # Load the English model with entity linking
#     nlp = spacy.load("en_core_web_sm")
#     nlp.add_pipe("entity_linker", config={"incl_prior": False})
#
#     # Process the text
#     doc = nlp(text)
#
#     # Print entities and their links
#     print("Entities and their links:")
#     for ent in doc.ents:
#         if ent.kb_id_:
#             print(f"{ent.text} ({ent.label_}) -> {ent.kb_id_} ({ent._.kb_qid})")
#
#             # Get the description of the linked entity
#             try:
#                 description = nlp.get_pipe("entity_linker").kb.get_vector(ent.kb_id_)["description"]
#                 print(f"  Description: {description}")
#             except KeyError:
#                 print("  No description available")
#         else:
#             print(f"{ent.text} ({ent.label_}) -> No link found")
#
#     # Custom logic for abbreviation expansion
#     words = text.split()
#     for i, word in enumerate(words):
#         if word.isupper() and len(word) > 1:
#             # Check if the next words form the full name
#             for j in range(i + 1, len(words)):
#                 if all(word.startswith(w[0].upper()) for w in words[i + 1:j + 1]):
#                     print(f"\nPossible abbreviation expansion: {word} -> {' '.join(words[i + 1:j + 1])}")
#                     break
#
#
# # Example usage
# text = "The NRA, which stands for the National Rifle Association, is a major organization in the United States. It has been involved in various campaigns. Donald Trump and Joe Biden have both mentioned the NRA in their speeches."
#
# perform_entity_linking(text)
#
# import spacy
# from collections import defaultdict
#
#
# def extract_and_link_entities(texts):
#     if isinstance(texts, str):
#         texts = [texts]
#
#     nlp = spacy.load("en_core_web_sm")
#
#     all_entities = defaultdict(set)
#     entity_mentions = defaultdict(list)
#
#     for text in texts:
#         doc = nlp(text)
#
#         for ent in doc.ents:
#             all_entities[ent.label_].add(ent.text)
#             entity_mentions[ent.text.lower()].append(ent.text)
#
#     # Perform entity linking
#     linked_entities = {}
#     for entity_type, entities in all_entities.items():
#         for entity in entities:
#             key = entity.lower()
#             if key not in linked_entities:
#                 linked_entities[key] = {
#                     'canonical': entity,
#                     'mentions': set(entity_mentions[key]),
#                     'type': entity_type
#                 }
#             else:
#                 linked_entities[key]['mentions'].update(entity_mentions[key])
#
#     # Link potential abbreviations
#     for key, entity in linked_entities.items():
#         words = entity['canonical'].split()
#         if len(words) > 1:
#             potential_abbr = ''.join(word[0].upper() for word in words)
#             if potential_abbr in all_entities[entity['type']]:
#                 abbr_key = potential_abbr.lower()
#                 if abbr_key in linked_entities:
#                     linked_entities[key]['mentions'].update(linked_entities[abbr_key]['mentions'])
#                     linked_entities[abbr_key] = linked_entities[key]
#
#     return linked_entities
#
#
# def print_linked_entities(linked_entities):
#     print("Linked Entities:")
#     for key, entity in linked_entities.items():
#         print(f"Canonical: {entity['canonical']} (Type: {entity['type']})")
#         print(f"  Mentions: {', '.join(entity['mentions'])}")
#         print()
#
#
# # Example usage
# texts = [
#     "The NRA, which stands for the National Rifle Association, is a major organization in the United States. It has been involved in various campaigns.",
#     "Donald Trump and Joe Biden have both mentioned the NRA in their speeches. Trump has been vocal about his support for the organization."
# ]
#
# linked_entities = extract_and_link_entities(texts)
# print_linked_entities(linked_entities)

import spacy


# def extract_and_link_entities(texts):
#     if isinstance(texts, str):
#         texts = [texts]
#
#     # Load the larger model with the entity linker
#     nlp = spacy.load("en_core_web_trf")
#
#     # Ensure the entity linker is in the pipeline
#     if "entity_linker" not in nlp.pipe_names:
#         nlp.add_pipe("entity_linker", last=True)
#
#     linked_entities = {}
#
#     for text in texts:
#         doc = nlp(text)
#
#         for ent in doc.ents:
#             key = ent.text.lower()
#             if key not in linked_entities:
#                 linked_entities[key] = {
#                     'canonical': ent.text,
#                     'mentions': set([ent.text]),
#                     'type': ent.label_,
#                     'kb_id': ent.kb_id_,
#                     'description': ''
#                 }
#             else:
#                 linked_entities[key]['mentions'].add(ent.text)
#
#             # If the entity is linked to a KB entry, get its description
#             if ent.kb_id_:
#                 try:
#                     linked_entities[key]['description'] = nlp.get_pipe("entity_linker").kb.get_vector(ent.kb_id_)[
#                         "description"]
#                 except KeyError:
#                     pass  # No description available
#
#     # Link entities with the same KB ID
#     kb_id_to_key = {}
#     for key, entity in linked_entities.items():
#         if entity['kb_id']:
#             if entity['kb_id'] in kb_id_to_key:
#                 # Merge this entity with the existing one
#                 existing_key = kb_id_to_key[entity['kb_id']]
#                 linked_entities[existing_key]['mentions'].update(entity['mentions'])
#                 del linked_entities[key]
#             else:
#                 kb_id_to_key[entity['kb_id']] = key
#
#     return linked_entities
#
#
# def print_linked_entities(linked_entities):
#     print("Linked Entities:")
#     for key, entity in linked_entities.items():
#         print(f"Canonical: {entity['canonical']} (Type: {entity['type']})")
#         print(f"  KB ID: {entity['kb_id']}")
#         print(f"  Mentions: {', '.join(entity['mentions'])}")
#         if entity['description']:
#             print(f"  Description: {entity['description']}")
#         print()
#
#
# # Example usage
# texts = [
#     "The NRA, which stands for the National Rifle Association, is a major organization in the United States. It has been involved in various campaigns.",
#     "Donald Trump and Joe Biden have both mentioned the NRA in their speeches. Trump has been vocal about his support for the organization."
# ]
#
# linked_entities = extract_and_link_entities(texts)
# print_linked_entities(linked_entities)

import spacy  # version 3.0.6'
import csv
from collections import defaultdict
# initialize language model
nlp = spacy.load("en_core_web_md")

# add pipeline (declared through entry_points in setup.py)
nlp.add_pipe("entityLinker", last=True)

# doc = nlp(text)

# returns all entities in the whole document
allowed_entity_types = ["EVENT", "FAC", "GPE", "LAW", "NORP", "ORG", "PERSON", "PER"]
entity_ob = defaultdict(list)
with open("abortion_arguments.tsv") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for i, line in enumerate(tsv_file):
        sent = line[0]
        # if i in [0, 24, 26, 200]:
        #     print(sent)

        doc = nlp(sent)
        # for ent in doc.ents:
        #     print(ent.text, ent.label_)  # Display the entity and its type (e.g., PERSON, ORG)
        main_entities = [ent.text for ent in doc.ents if ent.label_ in allowed_entity_types]
        main_entities = list(set(main_entities))
        for ent in main_entities:
            entity_ob[ent].append(i)
            # if ent in self.entity_ob:
            #     self.entity_ob[ent]['ids'] = [i]
            # else:
            #     self.entity_ob[ent] = {'ids': [i]}
    entity_linker_ob = defaultdict(list)
    for i, sent in enumerate(tsv_file):
        doc = nlp(sent)
        all_linked_entities = doc._.linkedEntities
        for ent in all_linked_entities:
            if ent.get_span() in entity_ob:
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
                entity_ob[first_text].extend(entity_ob[ent_text])
    count_edges = 0
    for entity_text in entity_ob:
        # print(f"entity_id: {entity_text}, entity: {entity_ob[entity_text]}")
        if len(entity_ob[entity_text]) > 1:
            c = len(entity_ob[entity_text]) * (len(entity_ob[entity_text]) + 1) // 2
            if c > 10:
                print(f"entity_text: {entity_text}, count: {c}")
            count_edges += c
    entity_count_per_sentence = defaultdict(int)
    for entity_text in entity_ob:
        # print(f"entity_id: {entity_text}, entity: {entity_ob[entity_text]}")
        if len(entity_ob[entity_text]) > 1:
            for sent_id in entity_ob[entity_text]:
                entity_count_per_sentence[sent_id] += 1
    entity_count_per_sentence_list = list(entity_count_per_sentence.items())
    entity_count_per_sentence_list.sort(key=lambda x: x[1], reverse=True)
    print(f"entity_count_per_sentence_list: {entity_count_per_sentence_list}")
    #     print("entity details .. ")
    #     main_entities = []
    #     for ent in doc.ents:
    #         print(ent.text, ent.label_)  # Display the entity and its type (e.g., PERSON, ORG)
    #
    #     all_linked_entities = doc._.linkedEntities
    #     for ent in all_linked_entities:
    #         # print(ent.get_id(), ent.get_span())
    #         if ent.get_id() not in entity_ob:
    #             entity_ob[ent.get_id()] = {'ids': [i], 'word': ent.get_span()}
    #         else:
    #             entity_ob[ent.get_id()]['ids'].append(i)
    #     if i == 5:
    #         break
    # for entity_id in entity_ob:
    #     print(f"entity_id: {entity_id}, entity_info: {entity_ob[entity_id]}")

    # printing data line by line
#     for line in tsv_file:
#         print(line)
# for sent in text_list:
#     for i, line in enumerate(tsv_file):
#         sent = line[0]
#         doc = nlp(sent)
#         all_linked_entities = doc._.linkedEntities
#         for ent in all_linked_entities:
#             print(ent.get_id(), ent.get_span())
#         if i == 0:
#             break
# iterates over sentences and prints linked entities
# for sent in doc.sents:
#     sent._.linkedEntities.pretty_print()

# import spacy_entity_linker
# print(spacy_entity_linker.__file__)
#
# import spacy
# from spacy_entity_linker import EntityLinker
#
#
# def extract_and_link_entities(texts):
#     if isinstance(texts, str):
#         texts = [texts]
#
#     # Load the model and add the entity linker
#     nlp = spacy.load("en_core_web_sm")
#     entity_linker = EntityLinker()
#     nlp.add_pipe(entity_linker, after="ner")
#
#     linked_entities = {}
#
#     for text in texts:
#         doc = nlp(text)
#
#         for ent in doc.ents:
#             key = ent.text.lower()
#             if key not in linked_entities:
#                 linked_entities[key] = {
#                     'canonical': ent.text,
#                     'mentions': set([ent.text]),
#                     'type': ent.label_,
#                     'kb_id': ent._.kb_id,
#                     'description': ent._.description if ent._.description else ''
#                 }
#             else:
#                 linked_entities[key]['mentions'].add(ent.text)
#
#             # If the entity is linked, update the KB ID and description
#             if ent._.kb_id:
#                 linked_entities[key]['kb_id'] = ent._.kb_id
#                 linked_entities[key]['description'] = ent._.description if ent._.description else ''
#
#     return linked_entities
#
#
# def print_linked_entities(linked_entities):
#     print("Linked Entities:")
#     for key, entity in linked_entities.items():
#         print(f"Canonical: {entity['canonical']} (Type: {entity['type']})")
#         print(f"  KB ID: {entity['kb_id']}")
#         print(f"  Mentions: {', '.join(entity['mentions'])}")
#         if entity['description']:
#             print(f"  Description: {entity['description']}")
#         print()
#
#
# # Example usage
# texts = [
#     "The NRA, which stands for the National Rifle Association, is a major organization in the United States. It has been involved in various campaigns.",
#     "Donald Trump and Joe Biden have both mentioned the NRA in their speeches. Trump has been vocal about his support for the organization."
# ]
#
# linked_entities = extract_and_link_entities(texts)
# print_linked_entities(linked_entities)

# print("...............")
# texts = ['Republicans always vote again abortion.', "As a strict Republican, I do not agree with that sentiment."]
# for s in texts:
#     doc = nlp(s)
#     all_linked_entities = doc._.linkedEntities
#     for ent in all_linked_entities:
#         print(ent.get_id(), ent.get_span())
#         # entity_linker_ob[ent.get_id()].append(ent.get_span())
