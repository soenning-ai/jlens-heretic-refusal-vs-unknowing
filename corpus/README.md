# Fit corpus

`prompts.json` holds the exact 150 prompts used to fit every lens in this repo —
English wikitext paragraphs of at least 128 Gemma tokens, no headings. It is shipped
verbatim so the fits are reproducible rather than approximately reproducible; the
selection is deterministic and `scripts/stage3_build_corpus.py` regenerates it.

**Source:** [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext),
config `wikitext-103-raw-v1`, split `train` — derived from Wikipedia and licensed
**CC BY-SA 3.0**. That license applies to the contents of this directory, not the
Apache-2.0 license that covers the rest of the repository.
