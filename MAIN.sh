export USER_ID=salesforce-b-wallace
export NUM_TPU=32
export TPU_NAME=sfr-${USER_ID}-tpu-${NUM_TPU}
export SAVE_DIR=/export/home/save/
sudo mkdir -p $SAVE_DIR && sudo chmod -R 777 $SAVE_DIR

python3 -m torch_xla.distributed.xla_dist --tpu=$TPU_NAME \
--restart-tpuvm-pod-server --env XLA_USE_BF16=0 \
-- python3 /export/home/diffusion/my_simclr_vit_profiling/run_example.py \
ckpt_dir=$SAVE_DIR batch_size=32 fake_data=True log_step_interval=50
