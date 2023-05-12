
ckpt=${1}

# below doesn't throw error and gives all
# could iterate through but would be slower b/c of model loading spin-up and need to figure out grouping together
# This is overall really annoying with quotes so hardcoding and moving on for now

python3 test_ckpt.py ${ckpt} --prompt "The tree of life made of cotton candy" "A goat enjoying himself at the state fair" "A triceratops police officer riding a rollercoaster" "A firefighter ostrich rescuing a chicken out of a burning building" "A rendering of the milky way galaxy" "A vegetable garden on the moon" --dim 512
