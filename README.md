This repo is based off of https://github.com/ronghanghu/moco_v3_tpu/tree/simclr_vit_release

The Salesforce TPU Research lab doc helps with setup for TPUs

Files of particular note:
* distributed.py (from the above repo) handles a lot of the Pytorch/XLA logic especially with respect to being amphibious with GPU
* run_example.py is the basic CLIP example
