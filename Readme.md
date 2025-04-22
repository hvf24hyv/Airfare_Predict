
# Table of Contents

1. [Airfare Prediction Using ML for US Domestic Travellers](#airfare-prediction-using-ml-for-us-domestic-travellers)
2. [Project Structure](#project-structure)
   - [Directory Overview](#directory-overview)
3. [Order of Running the Data Preparation Notebooks](#order-of-running-the-data-preparation-notebooks)
   - [Data Preparation Notebook](#data-preparation-notebook)
   - [Data Exploration Notebook](#data-exploration-notebook)
   - [Experiment Notebooks](#experiment-notebooks)
4. [Project Setup](#project-setup)
   - [Initial Setup](#initial-setup)
   - [Python Environment](#python-environment)
5. [Quick Commands to Push Changes to Git](#quick-commands-to-push-changes-to-git)

# Airfare prediction using ML for US domestic travellers

To build a data product that will help users in the USA to better estimate their local travel airfare. Users will be able to provide details of their trip and the app will predict the expected flight fare.

# Project Structure

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `_` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         Retail_Predict_Forecast and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── Airfare_Predict   <- Source code for use in this project.
    │
    ├── src/models             <- Stores the custom package codes 
    │
    ├── data_extraction.py     <- Contains custom fucntions to extract the data from the zipped raw data
    │
    ├── data_exploration.py    <- Contains custom fucntions to explore and understand the dataset
    │
    ├── null.py             <- Contains custom fucntions to create Null models for baseline
    │
    ├── performance.py      <- Code to run model inference with trained models          
    │   
    └── sets.py            <- Code to split the datasets and perform transformations on the features
```
--------
## Project setup
1. Go to the projects folder (Airfare_Predict)
2. pip install cookiecutter-data-science (give the 'Retail_Predict_Forecast as project and repo names)
3. ccds
4. cd Retail_Predict_Forecast 
5. initialise git repo 
	```bash  
    git init
6. login to git repo in a browser and create a repository with same <folder name>
7. In your local repo, link it with Github (replace the url with your username) -------> follow the steps given in git website
	```bash  
    git remote add origin git@github.com:<username>/<repo name>.git
8. Install Python version with Pyenv and set the local version
	```bash  
    pyenv install 3.11.4 and
    pyenv local 3.11.4
9. poetry init
	```bash  
    poetry add pandas numpy 
	poetry add jupyterlab==4.2.3 scikit-learn==1.5.1

10. To generate the requirements file
	```bash  
    poetry export --output requirements.txt 
11. To launch the poetry shell 
    ```bash  
    poetry shell 
12.  To launch the jupyter lab to run experiments with notebooks 
    ```bash  
    poetry run jupyter lab 

--------

--------

## Quick commands to push the changes to git
```bash 
git add .
git commit -m "message"
git push 

--------

