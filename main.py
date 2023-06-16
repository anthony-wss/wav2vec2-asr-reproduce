import argparse
import os
import json
from asr.preprocess import get_vocab_dict, preprocess_dataset
from asr.dataset import get_dataset, DataCollatorCTCWithPadding
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_metric
from asr.metric import ComputeMetrics
from transformers import TrainingArguments
import huggingface_hub
from transformers import Trainer

def main(args):
    if not os.path.exists(args.dict_path):
        vocab_dict = get_vocab_dict()
        print(vocab_dict)
        with open(args.dict_path, 'w') as f:
            json.dump(vocab_dict, f)
    huggingface_hub.login(open(args.passwd_file, 'r').readline().strip())
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    timit = get_dataset()
    timit = timit.map(preprocess_dataset, remove_columns=timit.column_names["train"], num_proc=4, fn_kwargs={'processor': processor})

    data_collator = DataCollatorCTCWithPadding(processor=processor)
    wer_metric = load_metric("wer")
    compute_metrics = ComputeMetrics(processor, wer_metric)

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base", 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir='wav2vec2-timit-asr',
        group_by_length=True,
        per_device_train_batch_size=32,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        gradient_checkpointing=True, 
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=timit["train"],
        eval_dataset=timit["test"],
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_path', type=str, default='./vocab.json')
    parser.add_argument('--passwd_file', type=str, default='./.write_token.passwd')
    args = parser.parse_args()
    main(args)