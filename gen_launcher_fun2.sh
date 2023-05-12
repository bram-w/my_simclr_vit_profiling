
ckpt=${1}

# below doesn't throw error and gives all
# could iterate through but would be slower b/c of model loading spin-up and need to figure out grouping together
# This is overall really annoying with quotes so hardcoding and moving on for now

python3 test_ckpt.py ${ckpt} --prompt "Mr Monopoly at a rodeo" "Snap crackle & pop playing the board game Sorry" "A ski run down to the Golden Gate Bridge" "A steampunk saloon" "A raccoon walking into a saloon" "A rubber duck hiding in the bushes" "Flower petals scattered across a mountain lake" "A rocketship taking off from stonehenge" "Rose petals on a path in a botanical garden" "The statue of a liberty as a cactus" "Pikachu hiking the Pacific Crest Trail" "A Squirtle getting married at the altar" --dim 512
