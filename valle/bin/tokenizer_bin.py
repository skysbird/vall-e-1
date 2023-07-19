#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Phonemize Text and EnCodec Audio.

Usage example:
    python3 bin/tokenizer.py \
        --src_dir ./data/manifests --output_dir ./data/tokenized

"""
import argparse
import logging
import os
from pathlib import Path

import torch
import torch.multiprocessing
from icefall.utils import get_executor
from lhotse import CutSet, NumpyHdf5Writer,load_manifest_lazy
from lhotse.recipes.utils import read_manifests_if_cached
from tqdm.auto import tqdm

from valle.data import (
    AudioTokenConfig,
    AudioTokenExtractor,
    TextTokenizer,
    tokenize_text,
    TtsDataModule
)
from valle.data.fbank import get_fbank_extractor
from valle.utils import SymbolTable

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Path to the manifest files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tokenized"),
        help="Path to the tokenized files",
    )
    parser.add_argument(
        "--text-extractor",
        type=str,
        default="espeak",
        help="espeak or pypinyin or pypinyin_initials_finals",
    )
    parser.add_argument(
        "--audio-extractor",
        type=str,
        default="Encodec",
        help="Encodec or Fbank",
    )
    parser.add_argument(
        "--dataset-parts",
        type=str,
        default="dev-clean test-clean",
        help="Space separated dataset parts",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="libritts",
        help="prefix of the manifest file",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="jsonl.gz",
        help="suffix of the manifest file",
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=400.0,
        help="The maximum number of audio seconds in a batch."
        "Determines batch size dynamically.",
    )

    return parser.parse_args()


def remove_short_and_long_utt(c):
    # Keep only utterances with duration between 1 second and 15.0 seconds
    #
    # Caution: There is a reason to select 15.0 here. Please see
    # ../local/display_manifest_statistics.py
    #
    # You should use ../local/display_manifest_statistics.py to get
    # an utterance duration distribution for your dataset to select
    # the threshold
    return c.duration <=10.0  or c.duration >= 20.0

#train_cuts = train_cuts.filter(remove_short_and_long_utt)


def main():
    args = get_args()
    args.manifest_dir = Path("data/tokenized")

    dataset_parts = args.dataset_parts.replace("--dataset-parts", "").strip()
    if dataset_parts == "all":  # LibriTTS
        dataset_parts = [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]

    #-p train -p dev -p test
    cn_dataset_parts = ['train','test','dev']

    assert len(dataset_parts) >= 1

    eng_manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=args.src_dir,
        prefix=args.prefix,
        suffix=args.suffix,
        types=["recordings", "supervisions", "cuts"],
    )

    #print(eng_manifests)

    cn_manifests = read_manifests_if_cached(
        dataset_parts=cn_dataset_parts,
        output_dir=args.src_dir,
        prefix="aishell",
        suffix=args.suffix,
        types=[ "recordings", "supervisions", "cuts"],
    )

    #print(args)


    #text_tokenizer = None
    text_tokenizer = None
    if args.text_extractor:
        text_tokenizer = TextTokenizer(backend=args.text_extractor,language='en-us')

    cn_text_tokenizer = TextTokenizer(backend="g2p_zh_en",language='zh-cn')

    audio_extractor = None
    if args.audio_extractor:
        if args.audio_extractor == "Encodec":
            audio_extractor = AudioTokenExtractor(AudioTokenConfig())
        else:
            assert args.audio_extractor == "Fbank"
            audio_extractor = get_fbank_extractor()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    unique_symbols = set()
    #num_jobs = min(32, os.cpu_count())
    num_jobs = 6
    #logging.info(f"dataset_parts: {dataset_parts} manifests {len(manifests)}")


    manifests_dict = {
            'libritts':eng_manifests,
            'aishell':cn_manifests
            }

    #pre process for token
    #dataset = TtsDataModule(args)

    #jsonl = [
    #    'libritts_cuts_dev-clean.jsonl.gz',
    #    'libritts_cuts_dev-other.jsonl.gz',
    #    'libritts_cuts_test-clean.jsonl.gz',
    #    'libritts_cuts_test-other.jsonl.gz',
    #    'libritts_cuts_train-clean-100.jsonl.gz',
    #    'libritts_cuts_train-clean-360.jsonl.gz',
    #    'libritts_cuts_train-other-500.jsonl.gz',
    #        ]

    #cut_set = load_manifest_lazy('data/tokenized/' + jsonl[0])
    #jsonl.pop(0)

    #for j in jsonl:
    #    tp = 'data/tokenized/'+j
    #    cut_set = cut_set + load_manifest_lazy(tp)

    # TextTokenizer
    #for c in tqdm(cut_set):
    #    lang = c.supervisions[0].language
    #    if lang == 'English':
    #        phonemes = tokenize_text(
    #            text_tokenizer, text=c.supervisions[0].text
    #        )
    #    elif lang == 'Chinese':
    #        phonemes = tokenize_text(
    #            cn_text_tokenizer, text=c.supervisions[0].text
    #        )
    #        c.supervisions[0].custom = {}
    #    c.supervisions[0].custom["tokens"] = {"text": phonemes}
    #    unique_symbols.update(phonemes)

    #print(unique_symbols)

    #prefix = args.prefix
    #临时修改两个语言的prefix,先写死
    #
    prefixs = ['libritts','aishell']
    #prefixs = ['aishell','libritts']
    #prefixs = ['wenetspeech']
    for prefix in prefixs:
        #if prefix and not prefix.endswith("_"):
        #    prefix = f"{prefix}_"
        manifests = manifests_dict[prefix]
        with get_executor() as ex:
            for partition, m in manifests.items():
                logging.info(
                    f"Processing partition: {partition} CUDA: {torch.cuda.is_available()}"
                )
                try:
                    cut_set = CutSet.from_manifests(
                        recordings=m["recordings"],
                        supervisions=m["supervisions"],
                    )

                except Exception as e:
                    cut_set = m["cuts"]

                # AudioTokenizer
                if args.audio_extractor:
                    if args.audio_extractor == "Encodec":
                        storage_path = (
                            f"{args.output_dir}/{prefix}_encodec_{partition}"
                        )
                    else:
                        storage_path = (
                            f"{args.output_dir}/{prefix}_fbank_{partition}"
                        )

                    if prefix.lower() in ["ljspeech", "aishell", "baker","wenetspeech"]:
                        cut_set = cut_set.resample(24000)
                        # https://github.com/lifeiteng/vall-e/issues/90
                        # if args.prefix == "aishell":
                        #     # NOTE: the loudness of aishell audio files is around -33
                        #     # The best way is datamodule --on-the-fly-feats --enable-audio-aug
                        #     cut_set = cut_set.normalize_loudness(
                        #         target=-20.0, affix_id=True
                        #     )

                    with torch.no_grad():
                        if (
                            torch.cuda.is_available()
                            and args.audio_extractor == "Encodec"
                        ):
                            cut_set = cut_set.compute_and_store_features_batch(
                                extractor=audio_extractor,
                                storage_path=storage_path,
                                num_workers=num_jobs,
                                batch_duration=args.batch_duration,
                                collate=False,
                                overwrite=True,
                                storage_type=NumpyHdf5Writer,
                            )
                        else:
                            cut_set = cut_set.compute_and_store_features(
                                extractor=audio_extractor,
                                storage_path=storage_path,
                                num_jobs=num_jobs if ex is None else 64,
                                executor=ex,
                                storage_type=NumpyHdf5Writer,
                            )

                # TextTokenizer
                for c in tqdm(cut_set):
                    lang = c.supervisions[0].language
                    if lang == 'English':
                        phonemes = tokenize_text(
                            text_tokenizer, text=c.supervisions[0].text
                        )
                    elif lang == 'Chinese':
                        phonemes = tokenize_text(
                            cn_text_tokenizer, text=c.supervisions[0].text
                        )
                        c.supervisions[0].custom = {}
                    c.supervisions[0].custom["tokens"] = {"text": phonemes}
                    unique_symbols.update(phonemes)

                #if args.text_extractor:
                #    if (
                #        args.prefix == "baker"
                #        and args.text_extractor == "labeled_pinyin"
                #    ):
                #        for c in tqdm(cut_set):
                #            phonemes = c.supervisions[0].custom["tokens"]["text"]
                #            unique_symbols.update(phonemes)
                #    else:
                #        for c in tqdm(cut_set):
                #            if args.prefix == "ljspeech":
                #                text = c.supervisions[0].custom["normalized_text"]
                #                text = text.replace("”", '"').replace("“", '"')
                #                phonemes = tokenize_text(text_tokenizer, text=text)
                #            elif args.prefix == "aishell":
                #                phonemes = tokenize_text(
                #                    text_tokenizer, text=c.supervisions[0].text
                #                )
                #                c.supervisions[0].custom = {}
                #            else:
                #                assert args.prefix == "libritts"
                #                phonemes = tokenize_text(
                #                    text_tokenizer, text=c.supervisions[0].text
                #                )
                #            c.supervisions[0].custom["tokens"] = {"text": phonemes}
                #            unique_symbols.update(phonemes)

                cuts_filename = f"{prefix}_cuts_{partition}.{args.suffix}"
                cut_set.to_file(f"{args.output_dir}/{cuts_filename}")

    #update symbo
    if args.text_extractor:
        unique_phonemes = SymbolTable()
        for s in sorted(list(unique_symbols)):
            unique_phonemes.add(s)
        logging.info(f"{len(unique_symbols)} unique phonemes: {unique_symbols}")

        unique_phonemes_file = f"{args.output_dir}/unique_text_tokens.k2symbols"
        unique_phonemes.to_file(unique_phonemes_file)


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
