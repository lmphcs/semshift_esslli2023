import torch
import warnings
from tqdm import tqdm


def collate(batch):
    return [
        {"input_ids": torch.cat([item[0]["input_ids"] for item in batch], dim=0),
         "attention_mask": torch.cat([item[0]["attention_mask"] for item in batch], dim=0)},
        [item[1] for item in batch],
        [item[2] for item in batch]
    ]


def get_context(tokenizer, token_ids, target_position, sequence_length):
    """
        Given a text containing a target word, return the sentence snippet which surrounds
        the target word (and the target word's position in the snippet).

        :param tokenizer: a Huggingface tokenizer
        :param token_ids: list of token ids for the entire context sentence
        :param target_position: tuple with the target word's start and end position in `token_ids`,
                                such that
                                token_ids[target_position[0]:target_position[1]] = target_word_ids
        :param sequence_length: desired length for output sequence (e.g., 128, 256, 512)
        :return: (context_ids, new_target_position)
                    context_ids: list of token ids in the target word's context window
                    new_target_position: tuple with the target word's start and
                    end position in `context_ids`
    """

    # -2 as [CLS] and [SEP] tokens will be added later; /2 as it's a one-sided window
    window_size = int((sequence_length - 4) / 2)
    target_len = target_position[1] - target_position[0]

    if target_len % 2 == 0:
        window_size_left = int(window_size - target_len / 2) + 1
        window_size_right = int(window_size - target_len / 2)
    else:
        window_size_left = int(window_size - target_len // 2)
        window_size_right = int(window_size - target_len // 2)

    # determine where context starts and if there are any unused context positions to the left
    if target_position[0] - window_size_left >= 0:
        start = target_position[0] - window_size_left
        extra_left = 0
    else:
        start = 0
        extra_left = window_size_left - target_position[0]

    # determine where context ends and if there are any unused context positions to the right
    if target_position[1] + window_size_right + 1 <= len(token_ids):
        end = target_position[1] + window_size_right + 1
        extra_right = 0
    else:
        end = len(token_ids)
        extra_right = target_position[1] + window_size_right + 1 - len(token_ids)

    # redistribute to the left the unused right context positions
    if extra_right > 0 and extra_left == 0:
        if start - extra_right >= 0:
            padding = 0
            start -= extra_right
        else:
            padding = extra_right - start
            start = 0
    # redistribute to the right the unused left context positions
    elif extra_left > 0 and extra_right == 0:
        if end + extra_left <= len(token_ids):
            padding = 0
            end += extra_left
        else:
            padding = end + extra_left - len(token_ids)
            end = len(token_ids)
    else:
        padding = extra_left + extra_right

    context_ids = token_ids[start:end]
    context_ids = [tokenizer.cls_token_id] + context_ids + [tokenizer.sep_token_id]
    item = {"input_ids": context_ids + padding * [tokenizer.pad_token_id],
            "attention_mask": len(context_ids) * [1] + padding * [0]}

    new_target_position = (target_position[0] - start + 1, target_position[1] - start + 1)

    return item, new_target_position


class ContextsDataset(torch.utils.data.Dataset):

    def __init__(self, targets_i2w, sentences, context_size, tokenizer, len_longest_tokenized=10,
                 n_sentences=None):
        super(ContextsDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.context_size = context_size

        with warnings.catch_warnings():
            for sentence in tqdm(sentences, total=n_sentences):
                sentence_token_ids_full = tokenizer.encode(' '.join(sentence),
                                                           add_special_tokens=False)
                sentence_token_ids = list(sentence_token_ids_full)
                while sentence_token_ids:
                    candidate_ids_found = False
                    for length in list(range(1, len_longest_tokenized + 1))[::-1]:
                        candidate_ids = tuple(sentence_token_ids[-length:])
                        if candidate_ids in targets_i2w:
                            sent_position = (len(sentence_token_ids) - length,
                                             len(sentence_token_ids))

                            context_ids, pos_in_context = get_context(
                                tokenizer, sentence_token_ids_full, sent_position, context_size)
                            self.data.append((context_ids, targets_i2w[candidate_ids],
                                              pos_in_context))

                            sentence_token_ids = sentence_token_ids[:-length]
                            candidate_ids_found = True
                            break
                    if not candidate_ids_found:
                        sentence_token_ids = sentence_token_ids[:-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        model_input, lemma, pos_in_context = self.data[index]
        model_input = {"input_ids": torch.tensor(
            model_input["input_ids"], dtype=torch.long).unsqueeze(0),
                       "attention_mask": torch.tensor(
                           model_input["attention_mask"], dtype=torch.long).unsqueeze(0)}
        return model_input, lemma, pos_in_context
