# Fold assignments

`step05b_primary_partitions.csv`: primary repeated nested CV (step 05b, tag `primary`). One row per patient, receptor, repeat and outer fold. `role = outer_test` rows give the outer fold in which the patient was held out; `role = outer_train` rows give the patient's inner fold within that outer fold's training set. Seed of repeat r is 42 + 1000·r; the inner splitter of outer fold f uses seed + f. Re-derived from the seeds by `src/analysis/step_23_partitions.py`; no model was refitted.

`step19_unitdemo_partitions.csv`: protocol analyses A1/A2 (lesion and case rows, random and patient-grouped arms), outer fold of every row as recorded by `src/analysis/step_19_unit_of_analysis.py`.
