#!/usr/bin/bash

# mount slce machine
# bash src/bash_scripts/mount_slce.sh

# Explicitly source pyenv's setup
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

echo "Beginning Pipeline"

######################## parameters ##########################
SUBSET=coris
MODALITY=Af
PID_COL=PID
LAT_COL=Laterality
DATE_COL=ExamDate
MODALITY_COL=Procedure
IMG_PATH_COL=file_path_coris
DATE=11182024
SEGDATE=12202024 #$DATE # by default
N_IMAGES=500
######################## parameters ##########################

#################### folders and files #######################

# project-level folders
GM_PROJECT_FOLDER=/sddata/projects/GA_progression_modeling
GM_DATASET_FOLDER=/sddata/data/GA_progression_modelling_data_redone
GA_SEG_PROJECT_FOLDER=/sddata/projects/GA_segmentation_Advaith/segmentation_generic/

# subfolders
CSV_FOLDER=data_csvs
RESULTS_FOLDER=results

# files
SEG_MODEL_FILES=(
"results/nkj5dgle_100/wandb-lightning/nkj5dgle/checkpoints/epoch=27-step=868.ckpt"
"results/ggloik75_100/wandb-lightning/ggloik75/checkpoints/epoch=29-step=930.ckpt"
"results/dx0nleu0_100/wandb-lightning/dx0nleu0/checkpoints/epoch=38-step=1209.ckpt"
"results/oiwciudt_100/wandb-lightning/oiwciudt/checkpoints/epoch=32-step=1023.ckpt"
"results/46i3qota_100/wandb-lightning/46i3qota/checkpoints/epoch=19-step=6280.ckpt"
"results/4r9cizei_100/wandb-lightning/4r9cizei/checkpoints/epoch=22-step=7222.ckpt"
"results/cahjqm4h_100/wandb-lightning/cahjqm4h/checkpoints/epoch=14-step=4710.ckpt"
"results/gtmxysh6_100/wandb-lightning/gtmxysh6/checkpoints/epoch=3-step=1256.ckpt"
)
CONFIGS=(
    "nopseudo_nohisteqtrain_nohisteqtest"
    "nopseudo_nohisteqtrain_histeqtest"
    "nopseudo_histeqtrain_nohisteqtest"
    "nopseudo_histeqtrain_histeqtest"
    "pseudo_nohisteqtrain_nohisteqtest"
    "pseudo_nohisteqtrain_histeqtest"
    "pseudo_histeqtrain_nohisteqtest"
    "pseudo_histeqtrain_histeqtest"
)
USE_HIST=(
    false
    true
    false
    true
    false
    true
    false
    true
)

#################### folders and files #######################

#################### run pipeline functions #######################

# get modality data
echo "1. Extracting modality specific data"
cd $GM_PROJECT_FOLDER || exit
pyenv activate ga_progression_modeling
mkdir $GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_${SUBSET}/segs_compare
src=$GM_DATASET_FOLDER/$CSV_FOLDER/$SUBSET/clean_data_${SUBSET}_${DATE}.csv
dst=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_${SUBSET}/segs_compare/clean_data_${SUBSET}_${DATE}_${MODALITY,,}.csv
if [ -e "$dst" ]; then
    echo ">> $dst exists."
else
    echo ">> $dst does not exist. Creating file..."
    python src/utils/get_modality_data.py \
        --csv "$src" \
        --modality_col "$MODALITY_COL" \
        --modality_val "$MODALITY" \
        --n_images "$N_IMAGES" \
        --save_as "$dst"
fi
printf "\n"

# get GA segmentations for that modality
for i in "${!SEG_MODEL_FILES[@]}"; do
    echo "2.$i. Extracting segmentations for data"
    cd $GA_SEG_PROJECT_FOLDER || exit
    pyenv activate ga_seg
    mkdir -p $GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_${SUBSET}/segs_compare/ga_segs_masks_${MODALITY,,}/$SEGDATE/
    src=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_${SUBSET}/segs_compare/clean_data_${SUBSET}_${DATE}_${MODALITY,,}.csv
    dst=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_${SUBSET}/segs_compare/ga_segs_masks_${MODALITY,,}/$SEGDATE/${CONFIGS[$i]}/
    if [ -e "$dst" ]; then
        echo ">> $dst exists."
    else
        echo ">> $dst does not exist. Creating segmentations..."
        python src/segmentation/generic/run/run_segs.py \
            --holdout_csv "$src" \
            --csv_img_path_col "$IMG_PATH_COL" \
            --hist_eq "${USE_HIST[$i]}" \
            --weights_path "${SEG_MODEL_FILES[$i]}" \
            --output_folder "$dst"
    fi
    printf "\n"

    # add paths to ga segs into dataset
    echo "3.$i. Adding GA segmentation paths to dataset"
    cd $GM_PROJECT_FOLDER || exit
    pyenv activate ga_progression_modeling
    src=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_${SUBSET}/segs_compare/clean_data_${SUBSET}_${DATE}_${MODALITY,,}.csv
    dst=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_${SUBSET}/segs_compare/clean_data_${SUBSET}_${DATE}_${MODALITY,,}_${CONFIGS[$i]}.csv
    python src/utils/add_seg_columns.py \
        --csv "$src" \
        --img_path_col "$IMG_PATH_COL" \
        --seg_path_col "file_path_ga_seg" \
        --modality_col "$MODALITY_COL" \
        --modality_val "$MODALITY" \
        --prefix "seg_" \
        --seg_folder "$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_${SUBSET}/segs_compare/ga_segs_masks_${MODALITY,,}/$SEGDATE/${CONFIGS[$i]}" \
        --save_as "$dst"
    printf "\n"
done

# create presentation comparing the segmentations
cd $GM_PROJECT_FOLDER || exit
echo "4. Creating presentation comparing GA segmentation models"
pyenv activate ga_progression_modeling
python src/ppt_generators/compare_ga_segs.py \
    --csv_dir $GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_${SUBSET}/segs_compare/ \
    --output_path $GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_${SUBSET}/segs_compare/segs_compare.pptx \
    --image_col "$IMG_PATH_COL" \
    --seg_col "file_path_ga_seg"

#################### run pipeline functions #######################

echo "Pipeline finished running!"