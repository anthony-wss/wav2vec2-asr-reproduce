from datasets import load_dataset
import re

symbol_to_remove = """[\,\?\.\!\-\;\:\"]"""

def normalize_text(batch):
    batch['text'] = re.sub(symbol_to_remove, '', batch['text']).lower()
    return batch

def extract_all_chars(batch):
    all_text = " ".join(batch['text'])
    vocab = list(set(all_text))
    return {'vocab': [vocab], 'all_text':[all_text]}

def get_vocab_dict():
    timit = load_dataset("timit_asr", data_dir='/tmp2/anthony/TIMIT')
    timit = timit.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])
    
    timit = timit.map(normalize_text)
    vocabs = timit.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=timit.column_names["train"])
    vocab_list = list(set(vocabs['train']['vocab'][0]))
    vocab_dict = {v: k for k,v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    return vocab_dict

def preprocess_dataset(batch, processor):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch
