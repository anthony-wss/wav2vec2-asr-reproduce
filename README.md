# Wav2Vec2

With only 10 minutes labeled dataset, a fine-tuned wav2vec2 model can achieve less than 5% WER on the clean test set of of LibirSpeech.

We do not implement language model in this implementation.

## Dataset

TIMIT with 5 hours of training data.

### Functions to preview

Preview full texts
```python
import pandas as pd
from IPython.display import display, HTML

sub_df = timit['train'].remove_columns(['file', 'audio'])
sub_df = pd.DataFrame(sub_df[:5])
display(HTML(sub_df.to_html()))
```

Previewing audio seems not working in vscode.

### Wav2Vec2Processor

Normally call `processor(...)` would be directed to `Wav2Vec2FeatureExtractor`.

If the call is in `with processor.as_target_processor():`, the call would be directed to `Wav2Vec2CTCTokenizer`

### Trainer arguments

`group_by_length`: can significantly speed up the training by grouping the inputs of similar length.

## Future work

- [ ] Add PER, CER metirc
- [ ] Add more upstream model
- [ ] Add more datasets
- [ ] Merge with language model during decoding

## Reference
* codes: https://huggingface.co/blog/fine-tune-wav2vec2-english
* ctc: https://distill.pub/2017/ctc/
