# Applications
Use this directory to run the simulated consortia tasks.

## Directories
* `cache`: used to store files

## Files
* `main_epsilon.py`: used to process the forecasting results and report metrics
* `forecast.py`: used to do the forecasting into the future based on an initial input. Currently saved to /work/zah8/forecasts
    * `future_outlook.py`: used to perform the forecast
* `future_v2.py`: used to train the regression models to take 128 time point as an input and predict the next 128 time points. Currently saved to /work/zah8/consortia_regression_models
* `segment_comparison.py`: used to compare different segment sizes. Was not used in the figures or paper, but was used to inform decisions about our approach to forecasting.
     * `main_delta.py`: used to generate data for the `segment_comparison.py` file
* `data.py`: used to load data for the tasks
* `config.py`: used to define which consortia to analyze
* `future_one_by_one.py`: used for some of the downstream tasks, primarily with the experimental consortia

## Regression model parameters
For the regression models, I trained them on 6 different training dataset sizes. For each training dataset, I used different parameters due to out of memory errors on the DCC. The smaller datasets were tested on a smaller model to see the robustness in a limited model, but this is not critical. Each of the training datasets were run with fivefold cross-validation, where 20% of the curves were randomly withheld.

* 8,000 curves: max_depth = 15
* 4,000 curves: max_depth = 15
* 1,000 curves: max_depth = 20
* 200 curves: max_depth = 3
* 80 curves: max_depth = 3
* 20 curves: max_depth = 3