
ckpt=${1}

# below doesn't throw error and gives all
# could iterate through but would be slower b/c of model loading spin-up and need to figure out grouping together
# This is overall really annoying with quotes so hardcoding and moving on for now

python3 test_ckpt.py ${ckpt} --prompt "Two blue buses are seen parked next to each other" "A woman staying dry from the rain and holding an umbrella" "A church with stained glass windows depicting a hamburger and french fries." "A photo of a confused grizzly bear in calculus class." "A car playing soccer, digital art." "Darth Vader playing with raccoon in Mars during sunset." "A cube made of porcupine" --dim 512
