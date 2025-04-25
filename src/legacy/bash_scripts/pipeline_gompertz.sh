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
SEGDATE=12202024_model_9 #$DATE # by default
USE_GOMPERTZ=true
FILTER_REGS=true
######################## parameters ##########################

#################### folders and files #######################

# project-level folders
GM_PROJECT_FOLDER=/sddata/projects/GA_progression_modeling

# subfolders
RESULTS_FOLDER=results
GOMPERTZ_PARAMS_FILE=/sddata/projects/Mathematical_Modeling/dataframes/mathematical_modeling_model_9_curated_parameters.csv
GOMPERTZ_PLOTS_FOLDER=/sddata/projects/Mathematical_Modeling/gompertz-curve-plots/MODEL9_PLOTS/

#################### folders and files #######################

#################### run pipeline functions #######################

if $USE_GOMPERTZ; then
    echo "6. Get gompertz data"

    # copy gompertz file to folder
    mkdir -p $GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/gompertz_data_${MODALITY,,}_${SEGDATE}/gompertz_plots/
    for file in $GOMPERTZ_PLOTS_FOLDER/*.png; do
        id_lat=${file##*-}
        id_lat=${id_lat%.png}
        new_file=gompertz-fit-$id_lat.png
        cp "$file" $GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/gompertz_data_${MODALITY,,}_${SEGDATE}/gompertz_plots/"$new_file"
        echo "$id_lat"
    done > $GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/specific_pats_${MODALITY,,}_${SEGDATE}.txt

    # copy parameters file
    cp $GOMPERTZ_PARAMS_FILE $GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/gompertz_data_${MODALITY,,}_${SEGDATE}/parameters.csv

    # merge metadata
    echo "7. Merge additional metadata"
    cd $GM_PROJECT_FOLDER || exit
    pyenv activate ga_progression_modeling
    src=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/area_comparisons_${MODALITY,,}_${SEGDATE}_wmetadata_affine.csv
    dst=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/area_comparisons_${MODALITY,,}_${SEGDATE}_wmetadata_affine.csv
    python src/utils/merge_metadata.py \
    --data_csv "$src" \
    --pid_col $PID_COL \
    --patinfo "/sddata/projects/GA_progression_modeling/data/GA_progression_modelling_data_redone/data_csvs/other/coris_patient_metadata.csv" \
    --registryinfo "data/GA_progression_modelling_data_redone/data_csvs/other/AMDDatabaseLogitudin_DATA_2023_08_17 for JK.csv" \
    --prsinfo "data/GA_progression_modelling_data_redone/data_csvs/other/amd_registry_prs_scores_23Jul2024.txt" \
    --gompertz "$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/gompertz_data_${MODALITY,,}_${SEGDATE}/parameters.csv" \
    --save_as "$dst"
    printf "\n"

    # create ppt file
    echo "8. Create ppt"
    cd $GM_PROJECT_FOLDER || exit
    pyenv activate ga_progression_modeling
    src=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/
    python src/ppt_generators/create_growth_curves_ppt_v2.py \
    --results-folder "$src" \
    --results-file "area_comparisons_${MODALITY,,}_${SEGDATE}_wmetadata_affine.csv" \
    --img-col $IMG_PATH_COL \
    --seg-col "file_path_ga_seg" \
    --patient-col "$PID_COL" \
    --laterality-col "$LAT_COL" \
    --date-col "$DATE_COL" \
    --area-manual-col "GA Size Final" \
    --area-ai-col "mm_area" \
    --perimeter-ai-col "mm_perimeter" \
    --n-foci-ai-col "n_foci" \
    --ppt-folder "powerpoint_${MODALITY,,}_${SEGDATE}_gompertz_affine" \
    --specific-pat specific_pats_${MODALITY,,}_${SEGDATE}.txt \
    --gompertz gompertz_data_${MODALITY,,}_${SEGDATE}/ \
    --filter_regs $FILTER_REGS
    printf "\n"

else

    # merge metadata
    echo "6. Merge additional metadata"
    cd $GM_PROJECT_FOLDER || exit
    pyenv activate ga_progression_modeling
    src=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/area_comparisons_${MODALITY,,}_${SEGDATE}_wmetadata.csv
    dst=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/area_comparisons_${MODALITY,,}_${SEGDATE}_wmetadata.csv
    python src/utils/merge_metadata.py \
    --data_csv "$src" \
    --pid_col $PID_COL \
    --patinfo /sddata/projects/GA_progression_modeling/data/GA_progression_modelling_data_redone/data_csvs/other/coris_patient_metadata.csv \
    --registryinfo "data/GA_progression_modelling_data_redone/data_csvs/other/AMDDatabaseLogitudin_DATA_2023_08_17 for JK.csv" \
    --prsinfo "data/GA_progression_modelling_data_redone/data_csvs/other/amd_registry_prs_scores_23Jul2024.txt" \
    --save_as "$dst"
    printf "\n"

    # create ppt file
    echo "7. Create ppt"
    cd $GM_PROJECT_FOLDER || exit
    pyenv activate ga_progression_modeling
    src=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/
    python src/ppt_generators/create_growth_curves_ppt_v2.py \
    --results-folder "$src" \
    --results-file "area_comparisons_${MODALITY,,}_${SEGDATE}_wmetadata.csv" \
    --img-col $IMG_PATH_COL \
    --seg-col "file_path_ga_seg" \
    --patient-col "$PID_COL" \
    --laterality-col "$LAT_COL" \
    --date-col "$DATE_COL" \
    --area-manual-col "GA Size Final" \
    --area-ai-col "mm_area" \
    --perimeter-ai-col "mm_perimeter" \
    --n-foci-ai-col "n_foci" \
    --ppt-folder "powerpoint_${MODALITY,,}_${SEGDATE}" \
    --filter_regs $FILTER_REGS
    printf "\n"
fi

#################### run pipeline functions #######################

echo "Pipeline finished running!"