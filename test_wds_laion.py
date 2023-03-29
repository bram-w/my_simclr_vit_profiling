import my_webdataset as wds
from my_webdataset import DataPipeline, WebLoader
import torchvision.transforms as T
import torch
import os
from glob import glob
data_dir = "/mnt/sfr-laion400m-data-ssd-pv-us-central1-a/laion115m_capfilt_20220817"
train_shards = glob(os.path.join(data_dir, "*", "*.tar"))

import transformers

model_name = 'CompVis/stable-diffusion-v1-4'
tokenizer = transformers.CLIPTokenizer.from_pretrained(
        model_name, subfolder="tokenizer", revision=None
    )

tokenizer_call = lambda s: tokenizer(s,
                    padding="max_length",
                                      max_length=tokenizer.model_max_length,
                                      truncation=True,
                                      return_tensors='pt').input_ids.squeeze()

viz_transform =     T.Compose(
                    [
                        T.RandomResizedCrop(224, scale=(0.5, 1.0)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                        )
local_batch_size = 2

train_dataset = DataPipeline(
         wds.ResampledShards(train_shards),
        # we now have an iterator over all shards
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.shuffle(10, handler=wds.warn_and_continue),
        wds.decode("pil", handler=wds.warn_and_continue),
        # we now have a list of decompressed train samples from each shard in this worker, in sequence
        wds.to_tuple("ppm;jpg;jpeg;png", "txt;json", handler=wds.warn_and_continue),
        wds.map_tuple(viz_transform, lambda x: x[0], # lambda s: torch.tensor(tokenizer_call(s)),
                    handler=wds.warn_and_continue),
        wds.batched(local_batch_size),
        )# .with_epoch(epoch_size).with_length(epoch_size) # adds `__len__` method to dataset
train_loader = WebLoader(train_dataset, num_workers=4,
            batch_size=None)

for a,b in train_loader:
    print(a,b)
    print(a.shape, b.shape)
    break
