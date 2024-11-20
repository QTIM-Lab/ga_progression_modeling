#!/usr/bin/bash

folder=interpatient_registration_results/10012024/
data=/sddata/projects/GA_progression_modeling/results/09172024_coris/area_comparisons_af.csv
# data=data/GA_progression_modelling_data_redone/clean_data_coris_09162024.csv

mkdir $folder
mkdir $folder/atlas
mkdir $folder/atlas/OD_atlas
mkdir $folder/atlas/OS_atlas
mkdir $folder/reg2atlas
mkdir $folder/reg2atlas/OD_images/
mkdir $folder/reg2atlas/OD_images/ga_registered/
mkdir $folder/reg2atlas/OD_images/images_registered/
mkdir $folder/reg2atlas/OD_images/vessels_registered/
mkdir $folder/reg2atlas/OS_images/
mkdir $folder/reg2atlas/OS_images/ga_registered/
mkdir $folder/reg2atlas/OS_images/images_registered/
mkdir $folder/reg2atlas/OS_images/vessels_registered/
mkdir $folder/heatmap

# python src/interpatient_registration/create_retina_atlas.py \
# --csv $data \
# --n 1 \
# --lat OD \
# --save-to $folder/atlas/OD_atlas \
# --use-cuda

# python src/interpatient_registration/create_retina_atlas.py \
# --csv $data \
# --n 1 \
# --lat OS \
# --save-to $folder/atlas/OS_atlas \
# --use-cuda

python src/interpatient_registration/interpatient_registration.py \
--csv $data \
--lat OD \
--atlas $folder/atlas/OD_atlas/reference_0.png \
--save-to $folder/reg2atlas/OD_images/ \
--use-cuda \
--specific-pat results/09172024_coris/specific_patients.txt

python src/interpatient_registration/interpatient_registration.py \
--csv $data \
--lat OS \
--atlas $folder/atlas/OS_atlas/reference_0.png \
--save-to $folder/reg2atlas/OS_images/ \
--use-cuda \
--specific-pat results/09172024_coris/specific_patients.txt

python src/interpatient_registration/make_heatmap.py --atlas $folder/atlas/OD_atlas/reference_0.png \
--lat OD \
--masks-folder $folder/reg2atlas/OD_images/ga_registered/ \
--save-to $folder/ \
--use-cuda

python src/interpatient_registration/make_heatmap.py --atlas $folder/atlas/OS_atlas/reference_0.png \
--lat OS \
--masks-folder $folder/reg2atlas/OS_images/ga_registered/ \
--save-to $folder/ \
--use-cuda