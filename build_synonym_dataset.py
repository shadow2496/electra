from model import tokenization
import json


def convert_synonym_json_to_dict(thesaurus_list):
    thesaurus_dict = {}
    for thesaurus in thesaurus_list:
        word = thesaurus['word'].lower()
        if len(word.split()) > 1:
            continue
        if word not in thesaurus_dict:
            thesaurus_dict[word] = []

        synonyms = thesaurus['synonyms']
        for synonym in synonyms:
            synonym = synonym.lower()
            if len(synonym.split()) > 1:
                continue
            if synonym not in thesaurus_dict:
                thesaurus_dict[synonym] = []
            # append word & synonym
            if synonym not in thesaurus_dict[word]:
                thesaurus_dict[word].append(synonym)
            if word not in thesaurus_dict[synonym]:
                thesaurus_dict[synonym].append(word)
    return thesaurus_dict


# Convert synonym dictionary to concatenated text for tokenization
def convert_synonym_dict_to_text(thesaurus_dict):
    text = ''
    for word, synonyms in thesaurus_dict.items():
        text += word + '_'
        for synonym in synonyms:
            text += synonym + '_'
        text += '/'
    return text


def build_token_synonym(text, vocab_file, do_lower_case):
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file,
        do_lower_case=do_lower_case)
    tokens = tokenizer.tokenize(text)

    token_synonym = {}
    for vocab in tokenizer.vocab.keys():
        token_synonym[vocab] = []
    src = True
    for token in tokens:
        if src:
            src_token = token
            src = False
        elif token == '_':
            continue
        elif token == '/':
            src = True
            continue
        else:
            # append synonym token to corresponding src_token
            if token not in token_synonym[src_token]:
                token_synonym[src_token].append(token)
            # reverse-wise appending
            if src_token not in token_synonym[token]:
                token_synonym[token].append(src_token)

    return token_synonym


if __name__ == '__main__':
    with open('./data/en_thesaurus.jsonl', 'r') as thesaurus_file:
        thesaurus_list = list(thesaurus_file)
        thesaurus_list = [json.loads(thesaurus) for thesaurus in thesaurus_list]

    thesaurus_dict = convert_synonym_json_to_dict(thesaurus_list)
    text = convert_synonym_dict_to_text(thesaurus_dict)
    token_synonym = build_token_synonym(text, './data/vocab.txt', True)
    synonym_json = json.dumps(token_synonym)
    with open('./data/synonym_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(synonym_json, f)
