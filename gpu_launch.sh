
# python3 run_example.py world_size=4 ckpt_dir=./tmp fake_data=False batch_size=64 device=cuda init_method=tcp://localhost:58472 iters_per_epoch=200 ckpt_epoch_interval=2 pretrained_unet=False log_step_interval=1

python3 run_example.py world_size=4 ckpt_dir=./tmp fake_data=True batch_size=64 device=cuda init_method=tcp://localhost:58472 iters_per_epoch=200 ckpt_epoch_interval=2 pretrained_unet=False log_step_interval=1 # data_dir='/export/laion2b/laion2B-en/data'


