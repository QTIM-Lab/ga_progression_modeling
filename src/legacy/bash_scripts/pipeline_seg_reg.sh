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
SEGDATE=12202024_model_9 #12022024 #$DATE # by default
HIST_EQ=true
REG_AFFINE_OR_TPS=affine
######################## parameters ##########################

#################### folders and files #######################

# project-level folders
GM_PROJECT_FOLDER=/sddata/projects/GA_progression_modeling
GM_DATASET_FOLDER=/sddata/data/GA_progression_modelling_data_redone
GA_SEG_PROJECT_FOLDER=/sddata/projects/GA_segmentation_Advaith/segmentation_generic/
VESSEL_SEG_PROJECT_FOLDER=/sddata/projects/GA_segmentation_Advaith/segmentation_generic/
REG_PROJECT_FOLDER=/sddata/projects/LightGlue/

# subfolders
CSV_FOLDER=data_csvs
SEG_FOLDER=segmentations
RESULTS_FOLDER=results

# files
model5=results/oiwciudt_100/wandb-lightning/oiwciudt/checkpoints/epoch=32-step=1023.ckpt # nopseudo_histeqtrain_histeqtest
# model6=results/0xhpwy7s_100/wandb-lightning/0xhpwy7s/checkpoints/epoch=38-step=12246.ckpt # pseudo_nohisteqtrain_nohisteqtest
# model8=results/lgrt7fjb_100/wandb-lightning/lgrt7fjb/checkpoints/epoch=34-step=10990.ckpt # pseudo_histeqtrain_histeqtest
# model9=results/v6062al1_100/wandb-lightning/v6062al1/checkpoints/epoch=25-step=8164.ckpt # pseudo_nohisteqtrain_histeqtest
# pseudolabel_model=data/output_logger/qhdxygp3_100/wandb-lightning/qhdxygp3/checkpoints/epoch=30-step=961.ckpt

SEG_MODEL_FILE=$model9
MANUAL_GA_AREAS_FILE="other/GA Combined Cohort 1-2 Database 3-10-24.xlsx"
#################### folders and files #######################

#################### run pipeline functions #######################

# # get modality data
# echo "1. Extracting modality specific data"
# cd $GM_PROJECT_FOLDER || exit
# pyenv activate ga_progression_modeling
# src=$GM_DATASET_FOLDER/$CSV_FOLDER/$SUBSET/clean_data_${SUBSET}_${DATE}.csv
# dst=$GM_DATASET_FOLDER/$SEG_FOLDER/$SUBSET/clean_data_${SUBSET}_${DATE}_${MODALITY,,}.csv
# if [ -e "$dst" ]; then
#     echo ">> $dst exists."
# else
#     echo ">> $dst does not exist. Creating file..."
#     python src/utils/get_modality_data.py \
#         --csv "$src" \
#         --modality_col "$MODALITY_COL" \
#         --modality_val "$MODALITY" \
#         --save_as "$dst"
# fi
# printf "\n"

# # get GA segmentations for that modality
# echo "2. Extracting segmentations for data"
# cd $GA_SEG_PROJECT_FOLDER || exit
# pyenv activate ga_seg
# src=$GM_DATASET_FOLDER/$SEG_FOLDER/$SUBSET/clean_data_${SUBSET}_${DATE}_${MODALITY,,}.csv
# dst=$GM_DATASET_FOLDER/$SEG_FOLDER/$SUBSET/ga_segs_masks_${MODALITY,,}/$SEGDATE/
# if [ -e "$dst" ]; then
#     echo ">> $dst exists."
# else
#     echo ">> $dst does not exist. Creating segmentations..."
#     python src/segmentation/generic/run/run_segs.py \
#         --holdout_csv "$src" \
#         --csv_img_path_col "$IMG_PATH_COL" \
#         --hist_eq "$HIST_EQ" \
#         --weights_path "$SEG_MODEL_FILE" \
#         --output_folder "$dst"
# fi
# printf "\n"

# # add paths to ga segs into dataset
# echo "2. Adding GA segmentation paths to dataset"
# cd $GM_PROJECT_FOLDER || exit
# pyenv activate ga_progression_modeling
# src=$GM_DATASET_FOLDER/$CSV_FOLDER/$SUBSET/clean_data_${SUBSET}_${DATE}.csv
# dst=$GM_DATASET_FOLDER/$CSV_FOLDER/$SUBSET/clean_data_${SUBSET}_${DATE}.csv
# python src/utils/add_seg_columns.py \
#     --csv "$src" \
#     --img_path_col "$IMG_PATH_COL" \
#     --seg_path_col "file_path_ga_seg" \
#     --modality_col "$MODALITY_COL" \
#     --modality_val "$MODALITY" \
#     --prefix "seg_" \
#     --seg_folder "$GM_DATASET_FOLDER/$SEG_FOLDER/$SUBSET/ga_segs_masks_${MODALITY,,}/$SEGDATE" \
#     --save_as "$dst"
# printf "\n"

# # get vessel segmentations for that modality
# echo "3. Generating vessel segmentation for dataset"
# cd $VESSEL_SEG_PROJECT_FOLDER || exit
# pyenv deactivate 
# conda activate Retina_Seg
# src=$GM_DATASET_FOLDER/$SEG_FOLDER/$SUBSET/clean_data_${SUBSET}_${DATE}_${MODALITY,,}.csv
# dst=$GM_DATASET_FOLDER/$SEG_FOLDER/$SUBSET/vessel_segs_masks/
# if [ -e "$dst" ]; then
#     echo ">> $dst exists."
# else
#     echo ">> $dst does not exist. Creating segmentations..."
#     # rm -r test_images test_output
#     # mkdir test_images test_output
#     # mkdir -p $dst
#     # python convert_images.py \
#         # --csv_path "$src" \
#         # --img_path_col "$IMG_PATH_COL" \
#         # --output_dir test_images
#     # python Code/seg_ensemble.py test_images/ test_output/ model/UNet_Ensemble 10
#     # mv test_output/* $dst
# fi
# printf "\n"

# # add paths to vessel segs into dataset
# echo "3. Adding vessel segmentation paths to dataset"
# cd $GM_PROJECT_FOLDER || exit
# pyenv activate ga_progression_modeling
# src=$GM_DATASET_FOLDER/$CSV_FOLDER/$SUBSET/clean_data_${SUBSET}_${DATE}.csv
# dst=$GM_DATASET_FOLDER/$CSV_FOLDER/$SUBSET/clean_data_${SUBSET}_${DATE}.csv
# python src/utils/add_seg_columns.py \
#     --csv "$src" \
#     --img_path_col "$IMG_PATH_COL" \
#     --seg_path_col "file_path_vessel_seg" \
#     --modality_col "$MODALITY_COL" \
#     --modality_val "$MODALITY" \
#     --prefix "" \
#     --seg_folder "$GM_DATASET_FOLDER/$SEG_FOLDER/$SUBSET/vessel_segs_masks/" \
#     --save_as "$dst"
# printf "\n"

# # get GA areas
# echo "4. Computing GA Areas"
# cd $GM_PROJECT_FOLDER || exit
# pyenv activate ga_progression_modeling
# src=$GM_DATASET_FOLDER/$CSV_FOLDER/$SUBSET/clean_data_${SUBSET}_${DATE}.csv
# dst=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/area_comparisons_${MODALITY,,}_${SEGDATE}.csv
# if [ -e "$dst" ]; then
#     echo ">> $dst exists."
# else
#     echo ">> $dst does not exist. Computing areas..."
#     python src/utils/area_comparison.py \
#         --image_data "$src" \
#         --img_col "$IMG_PATH_COL" \
#         --ga_col "file_path_ga_seg" \
#         --vessel_col "file_path_vessel_seg" \
#         --pid_col "$PID_COL" \
#         --laterality_col "$LAT_COL" \
#         --date_col "$DATE_COL" \
#         --modality "$MODALITY" \
#         --manual_area_data "$GM_DATASET_FOLDER/$CSV_FOLDER/$MANUAL_GA_AREAS_FILE" \
#         --save_as "$dst"
# fi
# printf "\n"

# register images
echo "5. Registering all modality images"
cd $REG_PROJECT_FOLDER || exit
pyenv activate eyeliner
# reg=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/registration_results_${MODALITY,,}_${SEGDATE}/results.csv
if [ $REG_AFFINE_OR_TPS == 'tps' ]; then 
    reg=$(find $GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET -maxdepth 1 -type d -name "registration_results_${MODALITY,,}_*_tps")/results.csv
else
    reg=$(find $GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET -maxdepth 1 -type d -name "registration_results_${MODALITY,,}_*_affine")/results.csv
fi

# if a registration file already exists, use that
if [ -e "$reg" ]; then
    echo ">> $reg exists. Adding reg paths to csv only."
    cd $GM_PROJECT_FOLDER || exit
    pyenv activate ga_progression_modeling
    src=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/area_comparisons_${MODALITY,,}_${SEGDATE}.csv
    dst=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/area_comparisons_${MODALITY,,}_${SEGDATE}_wmetadata_affine.csv
    python src/utils/add_reg_columns.py \
        --csv "$src" \
        --reg "$reg" \
        --save_as "$dst"
else
    src=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/area_comparisons_${MODALITY,,}_${SEGDATE}.csv
    if [ $REG_AFFINE_OR_TPS == 'tps' ]; then 
        dst=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/registration_results_${MODALITY,,}_${SEGDATE}/
        python src/sequential_registrator.py \
            --data "$src" \
            --mrn "$PID_COL" \
            --lat "$LAT_COL" \
            --sequence "$DATE_COL" \
            --input "$IMG_PATH_COL" \
            --vessel "file_path_vessel_seg" \
            --od "file_path_ga_seg" \
            --size 256 \
            --inp "vessel" \
            --reg2start \
            --reg_method "tps" \
            --lambda_tps 1.0 \
            --device "cuda:0" \
            --save "$dst"
    else
        dst=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/registration_results_${MODALITY,,}_${SEGDATE}_affine/
        python src/sequential_registrator.py \
            --data "$src" \
            --mrn "$PID_COL" \
            --lat "$LAT_COL" \
            --sequence "$DATE_COL" \
            --input "$IMG_PATH_COL" \
            --vessel "file_path_vessel_seg" \
            --od "file_path_ga_seg" \
            --size 256 \
            --inp "vessel" \
            --reg2start \
            --reg_method "affine" \
            --device "cuda:0" \
            --save "$dst"
    fi
    
    # copy registration results file
    cp "$dst"/results.csv \
    $GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/area_comparisons_${MODALITY,,}_${SEGDATE}_affine_wmetadata.csv
fi

printf "\n"

#################### run pipeline functions #######################

echo "Pipeline finished running!"