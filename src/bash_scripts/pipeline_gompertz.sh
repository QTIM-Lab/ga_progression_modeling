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
SEGDATE=12202024 #12022024 #$DATE # by default
######################## parameters ##########################

#################### folders and files #######################

# project-level folders
GM_PROJECT_FOLDER=/sddata/projects/GA_progression_modeling

# subfolders
RESULTS_FOLDER=results
USE_GOMPERTZ=false
GOMPERTZ_FOLDER=/sddata/projects/Mathematical_Modeling/gompertz-curve-plots/Dec16/gompertz-plots-dec-16

# files

#################### folders and files #######################

#################### run pipeline functions #######################

if $USE_GOMPERTZ; then
    # create specific pats file
    # List files in current directory and process paths
    ls $GOMPERTZ_FOLDER | while read file; do
        # Check if file contains both required components
        if [[ $file == *"gompertz-fit-"* && $file == *".png" ]]; then
            # Remove 'gompertz-fit-' prefix and '.png' suffix
            processed_name=$(echo "$file" | sed 's/gompertz-fit-//g' | sed 's/\.png//g')
            echo "$processed_name"
        fi
    done > $GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/specific_pats_${SEGDATE}.txt

    # copy gompertz file to folder
    mkdir $GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/gompertz_data_${MODALITY,,}
    mkdir $GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/gompertz_data_${MODALITY,,}/gompertz_plots/
    cp -r $GOMPERTZ_FOLDER/* $GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/gompertz_data_${MODALITY,,}_${SEGDATE}/gompertz_plots/

    # copy parameters file
    cp /sddata/projects/Mathematical_Modeling/participant_parameter_estimates.csv $GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/gompertz_data_${MODALITY,,}_${SEGDATE}/

    # merge metadata
    echo "7. Merge additional metadata"
    cd $GM_PROJECT_FOLDER || exit
    src=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/area_comparisons_${MODALITY,,}_wmetadata.csv
    dst=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/area_comparisons_${MODALITY,,}_wmetadata.csv
    python src/utils/merge_metadata.py \
    --data_csv "$src" \
    --pid_col $PID_COL \
    --patinfo "data/GA_progression_modelling_data_redone/data_csvs/other/AMDDatabaseLogitudin_DATA_2023_08_17 for JK.csv" \
    --prsinfo "data/GA_progression_modelling_data_redone/data_csvs/other/amd_registry_prs_scores_23Jul2024.txt" \
    --gompertz "$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/gompertz_data_${MODALITY,,}_${SEGDATE}/participant_parameter_estimates.csv" \
    --save_as "$dst"
    printf "\n"

    # create ppt file
    echo "8. Create ppt"
    cd $GM_PROJECT_FOLDER || exit
    src=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/
    python src/ppt_generators/create_growth_curves_ppt.py \
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
    --ppt-folder "powerpoint_${MODALITY,,}_${SEGDATE}_gompertz" \
    --specific-pat specific_pats.txt
    printf "\n"
else

    # merge metadata
    echo "7. Merge additional metadata"
    cd $GM_PROJECT_FOLDER || exit
    src=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/area_comparisons_${MODALITY,,}_${SEGDATE}_wmetadata.csv
    dst=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/area_comparisons_${MODALITY,,}_${SEGDATE}_wmetadata.csv
    python src/utils/merge_metadata.py \
    --data_csv "$src" \
    --pid_col $PID_COL \
    --patinfo "data/GA_progression_modelling_data_redone/data_csvs/other/AMDDatabaseLogitudin_DATA_2023_08_17 for JK.csv" \
    --prsinfo "data/GA_progression_modelling_data_redone/data_csvs/other/amd_registry_prs_scores_23Jul2024.txt" \
    --save_as "$dst"
    printf "\n"

    # create ppt file
    echo "8. Create ppt"
    cd $GM_PROJECT_FOLDER || exit
    src=$GM_PROJECT_FOLDER/$RESULTS_FOLDER/${DATE}_$SUBSET/
    python src/ppt_generators/create_growth_curves_ppt.py \
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
    --specific-pat specific_pats.txt
    printf "\n"
fi

#################### run pipeline functions #######################

echo "Pipeline finished running!"