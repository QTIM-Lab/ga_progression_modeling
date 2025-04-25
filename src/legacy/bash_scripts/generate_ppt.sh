#!/usr/bin/bash

python src/ppt_generators/create_growth_curves_ppt.py \
--results-folder results/10182024_talisa/ \
--results-file registration_results_af/results.csv \
--img-col file_path_coris \
--seg-col file_path_ga_seg \
--patient-col PID \
--laterality-col Laterality \
--date-col ExamDate \
--area-manual-col "GA Size Final" \
--area-ai-col mm_area \
--perimeter-ai-col mm_perimeter \
--n-foci-ai-col n_foci \
--ppt-folder powerpoint_af
# --specific-pat specific_pats.txt