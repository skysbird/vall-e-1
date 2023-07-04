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
        "--symbol1",
        type=Path,
        help="Path to the symobl files",
    )
    parser.add_argument(
        "--symbol2",
        type=Path,
        help="Path to the symobl files",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("merge.symbol"),
        help="Path to the symobl files",
    )
    

    return parser.parse_args()



def main():
    args = get_args()

    symbolFile1 = args.symbol1
    symbolFile2 = args.symbol2
    outputFile = args.output

    table1 = SymbolTable.from_file(symbolFile1)
    table2 = SymbolTable.from_file(symbolFile2)




    #merge
    unique_symbols = set()
    for idx in range(0,len(table1)):
        unique_symbols.add(table1[idx])


    for idx in range(0,len(table2)):
        unique_symbols.add(table2[idx])

    # print(len(unique_symbols))


    unique_phonemes = SymbolTable()
    for s in sorted(list(unique_symbols)):
        unique_phonemes.add(s)
    
    unique_phonemes_file = outputFile
    unique_phonemes.to_file(unique_phonemes_file)


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
