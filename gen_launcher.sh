
ckpt=${1}

# below doesn't throw error and gives all
# could iterate through but would be slower b/c of model loading spin-up and need to figure out grouping together
# This is overall really annoying with quotes so hardcoding and moving on for now

python3 test_ckpt.py ${ckpt} --prompt "A neon sign" "A painting of mountains" "A photo of a dog" "Sunrise over a city" --dim 512
