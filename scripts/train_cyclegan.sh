#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --qos=high
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=dlc-arceus,dlc-groudon,dlc-moltres
#SBATCH --mem-per-gpu=10G
#SBATCH --time=7-00:00:00
#SBATCH --container-mounts=/data/temporary:/data/temporary,/data/pa_cpgarchive:/data/pa_cpgarchive \
#SBATCH --container-image="dodrio1.umcn.nl#uokbaseimage/diag:tf2.8-pt1.10-v1"
#SBATCH --output=/home/lindastuder/logs/cyclegan-%j.out

/usr/local/bin/python3.9 -m pip install --upgrade pip
# 1. Install required Python packages
export SM_FRAMEWORK=tf.keras
pip3 install -U \
    image-classifiers \
    albumentations==1.2.0 \
    segmentation-models \
    tensorflow_probability==0.11.1

# 2. Set up working directory
WORKDIR="/home/user/source"
mkdir -p $WORKDIR
cd $WORKDIR || exit

# 3. Remove any old copies
rm -rf pathology-common pathology-fast-inference diag-models

# 4. Copy necessary folders (assuming you're running from a directory that contains them)
cp -r /data/temporary/linda/cyclegan_stain_transform/libs/pathology-common $WORKDIR/pathology-common
cp -r /data/temporary/linda/cyclegan_stain_transform/libs/pathology-fast-inference $WORKDIR/pathology-fast-inference
cp -r /data/temporary/linda/cyclegan_stain_transform/libs/diag-models $WORKDIR/diag-models

cp -r /data/temporary/linda/cyclegan_stain_transform/configs $WORKDIR/configs
ls -R $WORKDIR/configs

# Tweaked by Leander
cp -r /data/temporary/linda/cyclegan_stain_transform/libs/pathology-cyclegan-stain-transformation $WORKDIR/pathology-cyclegan-stain-transformation

# 5. Export PYTHONPATH
export PYTHONPATH="$WORKDIR/pathology-common:$WORKDIR/diag-models:$WORKDIR/pathology-fast-inference:$WORKDIR/pathology-cyclegan-stain-transformation:$PYTHONPATH"
echo $PYTHONPATH

# Train cyclegan
#python3 /home/user/source/pathology-cyclegan-stain-transformation/scripts/main.py \
#    --run_name "cyclegan_test" \
#    --output_dir "/data/temporary/linda/cyclegan_stain_transform/output" \
#    --param_file_path "/data/temporary/linda/cyclegan_stain_transform/configs/training_params/config_training_linda.yaml" \
#    --data_file_path "/data/temporary/linda/cyclegan_stain_transform/configs/data/dataset_config.yaml" \
#    --albumentations_path "/data/temporary/linda/cyclegan_stain_transform/configs/albumentations/config_albumentations_thomas.yaml"

# Train one cycleGAN per center
CENTERS=("AMC" "Charite_Berlin" "Leuven" "UMCU" "Vienna" "Emory" "RUMC")

# Define base paths
OUTPUT_DIR="/data/temporary/projects/pathology-kidney-cyclegan/reader-study"
BASE_CONFIG_DIR="$WORKDIR/configs"

# Loop over each center
for CENTER in "${CENTERS[@]}"; do
    echo "Running for center: $CENTER"

    # Construct the paths for the current center
    RUN_NAME="$CENTER"
    DATA_FILE_PATH="$BASE_CONFIG_DIR/data/reader-study-$CENTER.yaml"

    # Create the output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"

    # Execute the python script
    python3 $WORKDIR/pathology-cyclegan-stain-transformation/scripts/main.py \
        --run_name "$RUN_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --param_file_path "$BASE_CONFIG_DIR/training_params/config_training_linda.yaml" \
        --data_file_path "$DATA_FILE_PATH" \
        --albumentations_path "$BASE_CONFIG_DIR/albumentations/config_albumentations_thomas.yaml"
done