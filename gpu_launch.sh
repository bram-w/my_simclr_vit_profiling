# python3 run_example.py world_size=4 ckpt_dir=./tmp fake_data=True batch_size=8  device=cuda init_method=tcp://localhost:58472
python3 run_example.py world_size=4 ckpt_dir=./tmp fake_data=True batch_size=4  device=cuda init_method=tcp://localhost:58472 iters_per_epoch=3 ckpt_epoch_interval=2 pretrained_unet=True log_step_interval=1


# python3 run_example.py world_size=4 ckpt_dir=./tmp fake_data=True batch_size=4  device=cuda init_method=tcp://localhost:58472
