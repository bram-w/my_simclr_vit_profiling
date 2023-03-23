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
