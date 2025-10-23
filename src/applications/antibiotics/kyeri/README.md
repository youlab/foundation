# Applications
Use this directory to run the antibiotic tasks related to Carolyn's work.

## Dirs
* `cache`: used to store files


## Files
* `classify_antibiotic_treatment.py`: used to build classifiers for determining which type of antibiotic was used in an experiment based only on the OD600 readings.
* `regress_antibiotic_concentration.py`: used to build regression models for determining the concentration antibiotic used in an experiment based only on the OD600 readings.
* `data.py`: used to load the data
    * `config.py`: used to configure where the data is coming from and how it is loaded
    * `process_functions.py`: used to configure where the data is coming from and how it is loaded
* `write_antibiotic_summary.py`: used to summarize results from the tasks
