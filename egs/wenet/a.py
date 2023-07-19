from lhotse import CutSet,load_manifest_lazy
from lhotse.dataset.speech_recognition import K2SpeechRecognitionDataset
from torch.utils.data import DataLoader
import random

from lhotse.dataset import (
    CutConcatenate,
    DynamicBucketingSampler,
    PrecomputedFeatures,
    SingleCutSampler,
    SpecAugment,
)


path = "data/tokenized/cuts_dev.jsonl.gz"
cutset = load_manifest_lazy(path)
#cutset = CutSet.from_file(path).to_eager()

#cutset = cutset.shuffle()

#ids = list(cutset.ids)

#random.shuffle(ids)

#n = 1
#b = 0
#for i in cutset.ids:
#    print(i)
#    if i.startswith("BAC"):
#        b = b+1
#    n = n + 1
#    if n==100:
#        print(f"BAC = {b/n}")
#        n = 1
#        b = 0

from valle.data.dataset import SpeechSynthesisDataset


train_sampler = DynamicBucketingSampler(
                cutset,
                max_duration=80,
                shuffle=True,
                num_buckets=12,
                drop_last=True,
                buffer_size=500000,
            )
#



dset = SpeechSynthesisDataset(cutset,
                )
dloader = DataLoader(dset, sampler=train_sampler, batch_size=None, num_workers=1)
print(dloader)

for batch in dloader:
    print(batch)

#for c in cutset:
#    print(c.id)
#print(cutset.shuffle())

#print(train_sampler)
