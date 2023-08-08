# Beta version, use at your own risk!

import logging
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from helpers import collate, ContextsDataset
from gensim.models.word2vec import LineSentence
import torch
import pandas as pd
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModelForMaskedLM, AutoTokenizer

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--model",
        "-m",
        help="Path to a language  model (bert-base-cased"
             "is a possible option)",
        required=True,
    ),
    arg(
        "--output",
        "-o",
        help="Path to the output Numpy archive",
        default="token_embeddings.npz",
    )

    args = parser.parse_args()

    # Loading the test set
    graded = pd.read_csv("targets/english/graded_nopos.txt", sep="\t", header=None,
                         names=['word', 'truth'])
    print(graded)

    targets = defaultdict(list)
    with open("targets/english/target_forms_udpipe.csv", 'r', encoding='utf-8') as f_in:
        for line in f_in.readlines():
            line = line.strip()
            forms = line.split(',')
            if len(forms) > 1:
                for form in forms:
                    if form not in targets[forms[0]]:
                        targets[forms[0]].append(form)
            else:
                line = line.split('\t')
                targets[line[0]].append(line[0])

    n_target_forms = sum([len(vals) for vals in targets.values()])
    print(f"Target lemmas: {len(targets)}.")
    print(f"Target word forms: {n_target_forms}.")
    targetforms = [item for el in targets for item in targets[el]]

    # Loading the model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, never_split=targetforms)
    model = AutoModelForMaskedLM.from_pretrained(args.model, output_hidden_states=True)
    model.to(device)

    print(f"Tokenizer's added tokens:{tokenizer.get_added_vocab()}")

    # Embedding part: extracting token representations of the target words

    # Store vocabulary indices of target words
    targets_ids = defaultdict(lambda: dict())
    for lemma in targets:
        for form in targets[lemma]:
            targets_ids[lemma][form] = tokenizer.encode(form, add_special_tokens=False)

    assert n_target_forms == sum([len(vals) for vals in targets_ids.values()])

    # maps all forms' token ids to their corresponding lemma:
    ids2lemma = {}
    # maps every lemma to a list of token ids corresponding to all word forms:
    lemma2ids = defaultdict(list)
    len_longest_tokenized = 0

    for lemma, forms2ids in targets_ids.items():
        for form, form_id in forms2ids.items():

            # remove '▁' from the beginning of subtoken sequences for XLM-R:
            if len(form_id) > 1 and form_id[0] == 6:
                form_id = form_id[1:]

            if len(form_id) == 0:
                print(f'Empty string? Lemma: {lemma}\t'
                      f'Form:"{form}"\tTokenized: "{tokenizer.tokenize(form)}"')
                continue

            if len(form_id) == 1 and form_id[0] == tokenizer.unk_token_id:
                print(f'Tokenizer returns UNK for this word form. '
                      f'Lemma: {lemma}\tForm: {form}\tTokenized: {tokenizer.tokenize(form)}')
                continue

            if len(form_id) > 1:
                print(f'Word form split into subtokens. '
                      f'Lemma: {lemma}\tForm: {form}\tTokenized: {tokenizer.tokenize(form)}')

            ids2lemma[tuple(form_id)] = lemma
            lemma2ids[lemma].append(tuple(form_id))
            if len(tuple(form_id)) > len_longest_tokenized:
                len_longest_tokenized = len(tuple(form_id))

    # The COHA corpora
    corpus1 = "corpora/ccoha1.txt.gz"
    corpus2 = "corpora/ccoha2.txt.gz"

    sentences = LineSentence(corpus2)

    nSentences = 0
    target_counter = {target: 0 for target in lemma2ids}
    for sentence in sentences:
        nSentences += 1
        if nSentences % 10000 == 0:
            print(f"Processed {nSentences} lines")
        sentence_token_ids = tokenizer.encode(" ".join(sentence), add_special_tokens=False)

        while sentence_token_ids:
            candidate_ids_found = False
            for length in list(range(1, len_longest_tokenized + 1))[::-1]:
                candidate_ids = tuple(sentence_token_ids[-length:])
                if candidate_ids in ids2lemma:
                    target_counter[ids2lemma[candidate_ids]] += 1
                    sentence_token_ids = sentence_token_ids[:-length]
                    candidate_ids_found = True
                    break
            if not candidate_ids_found:
                sentence_token_ids = sentence_token_ids[:-1]

    print(f"Total usages: {sum(list(target_counter.values()))}")
    for lemma in target_counter:
        print(f"{lemma}: {target_counter[lemma]}")

    # Container for usages (usage matrix)
    usages = {
        target: np.empty((target_count, model.config.hidden_size))
        for (target, target_count) in target_counter.items()
    }

    # Iterate over sentences and collect representations
    nUsages = 0
    curr_idx = {target: 0 for target in target_counter}

    # The length of a token's entire context window:
    context_window = 128

    # The number of sentences processed at once by the LM:
    batch_size = 16

    dataset = ContextsDataset(ids2lemma, sentences, context_window, tokenizer,
                              len_longest_tokenized,
                              nSentences)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate)
    iterator = tqdm(dataloader, desc="Iteration")

    print("Start extracting embeddings...")
    for step, batch in enumerate(iterator):
        model.eval()
        batch_tuple = tuple()
        for t in batch:
            try:
                batch_tuple += (t.to(device),)
            except AttributeError:
                batch_tuple += (t,)

        batch_input_ids = batch_tuple[0]
        batch_lemmas, batch_spos = batch_tuple[1], batch_tuple[2]

        with torch.no_grad():

            outputs = model(**batch_input_ids)

            if device == "cuda":
                hidden_states = [el.detach().cpu().clone().numpy() for el in outputs.hidden_states]
            else:
                hidden_states = [el.clone().numpy() for el in outputs.hidden_states]

            # store usage tuples in a dictionary: lemma -> (vector, position)
            for b_id in np.arange(len(batch_lemmas)):
                lemma = batch_lemmas[b_id]
                layers = [layer[b_id, batch_spos[b_id][0]:batch_spos[b_id][1], :] for layer in
                          hidden_states]
                usage_vector = np.mean(layers, axis=0)
                if usage_vector.shape[0] > 1:
                    usage_vector = np.mean(usage_vector, axis=0)
                # Empty representation:
                if usage_vector.shape[0] == 0:
                    print(b_id)
                    print(lemma)
                    print(usage_vector)
                    usage_vector = np.zeros(usage_vector.shape[1])
                usages[lemma][curr_idx[lemma], :] = usage_vector
                curr_idx[lemma] += 1
                nUsages += 1

    iterator.close()
    print(f"Total embeddings: {nUsages}")
    np.savez_compressed(args.output, **usages)
