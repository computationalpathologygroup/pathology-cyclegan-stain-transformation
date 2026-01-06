#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --qos=low
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --exclude=dlc-arceus,dlc-groudon
#SBATCH --mem-per-gpu=10G
#SBATCH --time=7-00:00:00
#SBATCH --container-mounts=/data/temporary:/data/temporary,/data/pa_cpgarchive:/data/pa_cpgarchive \
#SBATCH --container-image="dodrio1.umcn.nl#uokbaseimage/diag:tf2.8-pt1.10-v1"
#SBATCH --output=/home/lindastuder/logs/cyclegan-apply-%j.out


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

# Tweaked by Leander
cp -r /data/temporary/linda/cyclegan_stain_transform/libs/pathology-cyclegan-stain-transformation $WORKDIR/pathology-cyclegan-stain-transformation

# 5. Export PYTHONPATH
export PYTHONPATH="$WORKDIR/pathology-common:$WORKDIR/diag-models:$WORKDIR/pathology-fast-inference:$WORKDIR/pathology-cyclegan-stain-transformation:$PYTHONPATH"
echo $PYTHONPATH

# 6. Apply cycleGAN
# Center name needs to be passed as argument ("AMC" "Charite_Berlin" "Leuven" "UMCU" "Vienna" "Emory" "RUMC")
CENTER="$1"

image_folder="/data/temporary/projects/pathology-kidney-diaggraft/reader-study-preprocessing/images/$CENTER/wsi_tif"
mask_folder="/data/temporary/projects/pathology-kidney-diaggraft/reader-study-preprocessing/images/$CENTER/tissue_masks"
output_folder="/data/temporary/projects/pathology-kidney-diaggraft/reader-study-preprocessing/images/$CENTER/wsi_tif_cyclegan"
mkdir -p "$output_folder"

for image in "$image_folder"/*.tif; do
  image_name=$(basename "$image" .tif)
  mask_path="$mask_folder/${image_name}_tissue_mask.tif"
  output_path="$output_folder/${image_name}.tif"

  python3 /home/user/source/pathology-fast-inference/scripts/applygan_multiproc.py \
    --model_path "/data/temporary/projects/pathology-kidney-cyclegan/reader-study/$CENTER/checkpoint/source_to_target_epoch49.h5" \
    --input_wsi_path "$image" \
    --mask_wsi_path "$mask_path" \
    --output_wsi_path "$output_path" \
    --cache_directory "/home/user/cache" \
    --work_directory "/home/user/work/" \
    --read_spacing 0.5 \
    --mask_spacing 2.0 \
    --write_spacing 0.25 \
    --axes_order "whc" \
    --tile_size 512 \
    --readers 2 \
    --writers 2
done