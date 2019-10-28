import os

if os.path.exists("./cropped")==False:
    os.makedirs("./cropped")

ALPHA = 0.5
THRESHOLD = 0.5
IMAGE_SIZE= 96
LAYERS_TO_FREEZE= 30
NUM_EPOCHS= 10
STEPS_PER_EPOCH= 5
BATCH_SIZE= 64