# Tourism Forecasting
This repository contains the code and data used for the tourism forecasting competition organised by the School of Hospitality and Tourism Management, University of Surrey. The aim of this tourism forecasting competition is three-fold:
- To further advance the methodology of tourism forecasting in a crisis, particularly for demand recovery, and contribute to the development of this field of research;
- To inform the tourism industry and destination management and marketing organisations of the good forecasting practice and the predicted recovery of the Chinese outbound market;
- To promote the Curated Collection on Tourism Demand Forecasting of Annals of Tourism Research as a leading and main outlet for state-of-the-art tourism forecasting research.

The most up-to-date historical data (up to February 2023) of Chinese outbound tourist arrivals in 20 selected destinations will be provided to each team. These series will be forecast for this competition. The forecasts for the period from **August 2023 to July 2024** will be used for competition evaluation.

## Getting Started

To run the Jupyter notebooks, first setup the python environment as follow:
```bash
conda env create -f env.yml
conda activate tourism
```
OR, install the package to an existing environment
```bash
conda activate <your_env_name>
pip install -r requirements.txt
```
Setup JupyterLab so that the kernel appears in JupyterLab (optional)
```bash
python -m ipykernel install --user
```

## Documentation
See `docs` for the full report.

## File and Folder Structure

- data: Stores all raw, processed and imputed data.
- src: Source codes for python codes

**Files**
- **1.0_data_currency_scraper.ipynb**: Notebook for scraping historical currency rate from fxtop.com
- **1.1_data_visualization.ipynb**: EDA of tourism arrival data
- **1.2_data_downloaded.ipynb**: Processing and EDA of downloaded explanatory data
- **1.3_data_processing.ipynb**: Process and join all data into one table per destination
- **2.0_climate_forecasting.ipynb**: Notebook for forecasting future climate variables with SARIMA
- **2.1_fsi_forecasting.ipynb**: Notebook for forecasting FSI using other explanatory variables
- **2.2_fx_forecasting.ipynb**: Notebook for forecasting currency rate using other explanatory variables
- **2.3_visitor_impute.ipynb**: Notebook for imputing missing tourism arrival data with SARIMA
- **3.0_visitor_forecasting_precovid.ipynb**: Notebook for experiment involving only precovid data (for hyperparameter tuning and sanity check)
- **3.0a_visitor_forecasting_precovid_ablation.ipynb**: Notebook for ablation study of encoding methods for temporal feature
- **3.1_visitor_forecasting_covid**: Notebook for experiment involving training data up to 2022 and using 2022 as testing period. Also contains code to produce the final required forecast.