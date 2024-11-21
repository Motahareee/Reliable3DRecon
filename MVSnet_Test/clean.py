import os
import shutil

input_dir = "mvs_testing"
datasets = "handshake_phase2_100frames"
#datasets = "handshake"

frames = os.listdir(os.path.join(input_dir, datasets))

for frame in frames:

    if ".txt" in frame:
        continue

    files = os.listdir(os.path.join(input_dir, datasets, frame))

    for file in files:
        if ".jpg" in file:
            continue

        try:
            os.remove(os.path.join(input_dir, datasets, frame, file))
        except OSError as e:
            shutil.rmtree(os.path.join(input_dir, datasets, frame, file))

        #os.remove(os.path.join(input_dir, datasets, frame, file))
        #shutil.rmtree(os.path.join(input_dir, datasets, frame, file))

print("done")