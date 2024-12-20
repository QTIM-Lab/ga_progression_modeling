# GA Progression Modeling and Analysis using Deep Learning 

![alt](assets/pipeline.png)

This repo contains all code required to execute our GA progression monitoring pipeline from FAF imaging.

## Dependencies

Make sure you have pyenv and poetry setup to run code. Clone the repo and execute the following:

```
pyenv virtualenv 3.10.4 ga_progression_modeling
poetry install
```

## Libraries Used

Most of the code in this repo is to perform analysis of GA data, particularly calculating GA areas and analyzing change in are from GA segmentations and registering images of GA. The segmentation and registration libraries used are linked below. These need to be cloned and run separately.

1. Image registration using [EyeLiner](https://github.com/QTIM-Lab/EyeLiner)
2. GA image segmentation using [segmentation_generic](https://github.com/QTIM-Lab/ga_segmentation)
3. Retinal vessel segmentation using [vineet1992's](https://github.com/QTIM-Lab/retina_seg) vessel segmentation repo.

## Run Pipelines for GA Progression Modeling

There are multiple pipelines involved to create the GA progression model data, which are saved as `src/bash_scripts/pipeline_*.sh`.

### Setup

Before running these pipelines, there are some repos that you need to setup:

1. Setup GA Seg repo and train a GA Seg Model: Clone the [GA segmentation repo](https://github.com/QTIM-Lab/ga_segmentation) and follow all instructions in the README steps up until step 4.5. Step 4.5 in the github performs inference using a trained model, which you don't need to do, all we need is the weights and inference will happen inside this repo.

2. Setup Vessel Segmentation repo: Clone the [Vessel segmentation repo](https://github.com/QTIM-Lab/retina_seg) and follow the setup instructions in the README.

3. Setup EyeLiner repo: Clone the [EyeLiner repo](https://github.com/QTIM-Lab/EyeLiner) and follow the setup instructions.

### Pipelines Overview

After setting up the above, run the below pipelines in order by simply running `bash *pipeline_script*`. You will need to modify the configs inside these bash files:

```bash
Expected folder structure:
- GM_PROJECT_FOLDER (this repo)
    - GM_DATASET_FOLDER
        - CSV_FOLDER
            - SUBSET
                - clean_data_SUBSET_DATE.csv
        - SEG_FOLDER
            - SUBSET
                - ga_segs_masks_MODALITY
                    - SEGDATE
                - vessel_segs_masks
                - clean_data_SUBSET_DATE_MODALITY.csv
    - RESULTS_FOLDER
        - SUBSET_DATE
            - area_comparisons_MODALITY_DATE.csv
            - area_comparisons_MODALITY_wmetadata_DATE.csv
            - registration_results_MODALITY_DATE/
            - gompertz_data_MODALITY/
            
- GM_SEG_PROJECT_FOLDER
- VESSEL_SEG_PROJECT_FOLDER
- REG_PROJECT_FOLDER

######################## parameters ##########################
SUBSET=patient_cohort_name (can be set to anything)
MODALITY=image modality (for now we assume it is always Af - autofluorescence)
PID_COL=column indicating patient id
LAT_COL=column indicating eye laterality (OD/OS)
DATE_COL=column indicating date of image
MODALITY_COL=column indicating modality
IMG_PATH_COL=column indicating image path
DATE=date of progression modeling experiment (for version control)
SEGDATE=date of segmentation (for version control if training multiple segmentation models)
######################## parameters ##########################

#################### folders and files #######################

# project-level folders
GM_PROJECT_FOLDER=absolute path for this repo
GM_DATASET_FOLDER=absolute path to the folder containing the dataset csv
GA_SEG_PROJECT_FOLDER=absolute path to the GA segmentation repo
VESSEL_SEG_PROJECT_FOLDER=absolute path to the vessel segmentation repo
REG_PROJECT_FOLDER=absolute path to the image registration repo

# subfolders
CSV_FOLDER=csv folder
SEG_FOLDER=segmentations folder
RESULTS_FOLDER=results folder

# files
SEG_MODEL_FILE=path to trained seg model weights
HIST_EQ=additional parameters for seg model script
MANUAL_GA_AREAS_FILE=additional metadata files
#################### folders and files #######################
```

#### 1. `src/bash_scripts/pipeline_seg_reg.sh`

1. Extracts the longitudinal images for the specified modality from the input csv.

2. GA Segmentation: Providing your trained segmentation model in the bash script in `SEG_MODEL_FILE`, the script will redirect to the segmentation repo and run the model on all the modality images.

3. Vessel Segmentation: The script will redirect to the vessel segmentation repo then perform vessel segmentation inference on all the modality images.

4. GA Area Calculation: Areas of the GA are computed from the segmentation in physical units. Additionally, the scripts calculate perimeter, number of foci which may be useful for analysis.

5. Longitudinal Image Registration: The script will implement Eyeliners sequential registration framework to register all images of a patient.

The output file is a file called `area_comparisons_MODALITY_SEGDATE_wmetadata.csv` stored in the results folder. This contains all the image paths and segmentations, quantitative measures of the segmentations, and registration parameters. This will be utilized in the next pipeline.

#### 2. Perform Gompertz Analysis on the output csv produced from step 1.1. (To be done separately using Aaron Beckwith's code)

#### 3. `src/bash_scripts/pipeline_gompertz.sh` (To be updated...)

1. Joins gompertz data from pipeline 2 with the output csv from pipeline 1.
2. Joins additional metadata such as patient dob, PRS codes etc.
3. Creates a final powerpoint of the results.

### Internal Use for Lab Members

For folks in the lab, if running on the PowerEdge machine (big GPU), you can just run my bash pipelines directly with minimal changes. If running on any other server, you'll need to setup the repos as mentioned above. The datasets for my experiments are available here:

1. Progression modeling dataset: `/sddata/projects/GA_progression_modeling/data/GA_progression_modelling_data_redone`
2. GA segmentor training dataset: `/sddata/projects/GA_progression_modeling/data/geographic_atrophy/ga_and_nonga_110/csvs`