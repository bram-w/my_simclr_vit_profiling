
6/12/23

below runs fast on tPU

```bash
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export TPU_NUM_DEVICES=4
export ALLOW_MULTIPLE_LIBTPU_LOAD=1

export USER_ID=salesforce-b-wallace
export NUM_TPU=512
export TPU_NAME=sfr-${USER_ID}-tpu-${NUM_TPU}-B
export SAVE_DIR=gs://sfr-tpu-us-east1-research/bwallace/ckpts/models/sd


# Could do clever thing here of one epoch restart if something doesn't exist yadda yadda but for now just dropping resume_ckpt_path since I got it already
XLA_UNEVEN_HEARTBEAT_TIMEOUT=40000  XLA_EVEN_HEARTBEAT_TIMEOUT=40000 python3 -m torch_xla.distributed.xla_dist --tpu=$TPU_NAME \
--restart-tpuvm-pod-server --env XLA_USE_BF16=0 \
-- python3 /export/home/diffusion/my_simclr_vit_profiling/run_example.py \
ckpt_dir=$SAVE_DIR batch_size=2048 accumulate_grad_iter=4  fake_data=False log_step_interval=100 pretrained_unet=False ckpt_prefix=5_4_23_v4-128_bs_2048_lr_1e-4_aesthetics_512x512_after_19_epochs_base ckpt_epoch_interval=1  lr=1e-4 warmup_steps=40000 num_epochs=32 iters_per_epoch=12500 data_dir=/mnt/sfr-laion400m-data-ssd-pv-us-central1-a/laion-aesthetic/laion-aesthetic-v1/laion-aesthetic-en-51m/images num_workers=16 image_dim=512 override_opt_lr=True
```



----Below is older (still possibly relevant) notes -----
Startup command:
```bash
export NUM_TPU=32                
export USER_ID=salesforce-b-wallace 
export ZONE=us-east1-d         
export ACCELERATOR_TYPE=v3-${NUM_TPU}
export TPU_NAME=sfr-${USER_ID}-tpu-${NUM_TPU}
export RUNTIME_VERSION=tpu-vm-pt-1.10         
export PROJECT_ID=salesforce-research-internal


gcloud alpha compute tpus tpu-vm create ${TPU_NAME} --zone ${ZONE} \
    --reserved --accelerator-type ${ACCELERATOR_TYPE} --version ${RUNTIME_VERSION} \
    --network=sf-research-pv-network --subnetwork=sf-research-us-east1 --tags=tpu-vm \
    --metadata startup-script='#! /bin/bash
pip install braceexpand
pip install timm
pip install google-cloud-storage
pip install tensorboardX
pip install ftfy
pip install regex
pip3 install https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/wheels/libtpu-nightly/libtpu_nightly-0.1.dev20211015-py3-none-any.whl
apt-get update
apt-get install nfs-common -y
mkdir -p /export/home
mount 172.24.197.2:/sfr_home/b-wallace /export/home
echo "172.24.197.2:/sfr_home/b-wallace /export/home nfs rw 0 0"
EOF'
```

To run:
* install.sh
* MAIN.sh
