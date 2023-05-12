# Doing generations

Copy a file from the GCS bucket using `mkdir sd_ckpts; gsutil cp <gs-url> sd_ckpts/` then run prompts as `python3 test_ckpt.py ${ckpt} --prompt "A neon sign" "A painting of mountains" "A photo of a dog" "Sunrise over a city"`. See the `gen*.sh` scripts for example prompts. The dataset ablation models should be run @ 256 resolution (default})
