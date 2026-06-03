import os
import zipfile
import urllib
import numpy as np
import tiktoken


def get_wikitext_data(datasets_base_dir):
    """Inspired from https://github.com/tysam-code/hlb-gpt"""
    WIKITEXT_DATA_PATH = os.path.join(datasets_base_dir, "wikitext/")
    if not os.path.exists(WIKITEXT_DATA_PATH):
        os.makedirs(WIKITEXT_DATA_PATH, exist_ok=True)
        print("downloading data and tokenizing (1-2 min)")
        # original metamind S3 url is dead (301); Smerity mirror serves the same zip but
        # blocks the default Python User-Agent (403), so send a browser UA.
        raw_data_source = "https://wikitext.smerity.com/wikitext-103-raw-v1.zip"
        _req = urllib.request.Request(raw_data_source, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(_req) as _resp, open(
            os.path.join(WIKITEXT_DATA_PATH, "data.zip"), "wb"
        ) as _f:
            _f.write(_resp.read())

        with zipfile.ZipFile(
            os.path.join(WIKITEXT_DATA_PATH, "data.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(WIKITEXT_DATA_PATH)

        with open(
            os.path.join(WIKITEXT_DATA_PATH, "wikitext-103-raw/wiki.train.raw"), "r"
        ) as data_file:
            raw_train_data = data_file.read()

        with open(
            os.path.join(WIKITEXT_DATA_PATH, "wikitext-103-raw/wiki.valid.raw"), "r"
        ) as data_file:
            raw_eval_data = data_file.read()

        tokenizer = tiktoken.get_encoding("gpt2")
        raw_tokenized_train = tokenizer.encode_ordinary(raw_train_data)
        raw_tokenized_eval = tokenizer.encode_ordinary(raw_eval_data)

        train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16)
        eval_tokenized = np.array(raw_tokenized_eval, dtype=np.uint16)

        train_tokenized.tofile(os.path.join(WIKITEXT_DATA_PATH, "train.bin"))
        eval_tokenized.tofile(os.path.join(WIKITEXT_DATA_PATH, "val.bin"))
        print("completed the tokenization process!")

    return {
        "train": os.path.join(WIKITEXT_DATA_PATH, "train.bin"),
        "val": os.path.join(WIKITEXT_DATA_PATH, "val.bin"),
    }
