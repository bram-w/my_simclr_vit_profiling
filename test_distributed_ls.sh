export USER_ID=salesforce-b-wallace
export NUM_TPU=32
export TPU_NAME=sfr-${USER_ID}-tpu-${NUM_TPU}

python3 -m torch_xla.distributed.xla_dist --tpu=$TPU_NAME \
--restart-tpuvm-pod-server --env XLA_USE_BF16=0 \
-- ls /export/home/diffusion/my_simclr_vit_profiling/ -d


