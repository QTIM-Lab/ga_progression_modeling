# GA Progression Modeling and Analysis using Deep Learning 

## Dependencies

Make sure you have pyenv and poetry setup to run code. Clone the repo and execute the following:

```
pyenv virtualenv 3.10.4 ga_progression_modeling
poetry install
```

## Libraries Used

Most of the code in this repo is to perform analysis of GA data, particularly calculating GA areas and analyzing change in are from GA segmentations and registering images of GA. The segmentation and registration libraries used are linked below. These need to be cloned and run separately.

1. Image registration using EyeLiner: https://github.com/QTIM-Lab/EyeLiner
2. Image segmentation using segmentation_generic: https://github.com/QTIM-Lab/segmentation_generic