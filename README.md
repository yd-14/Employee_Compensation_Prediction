### Step 1: Create a new environment

```
conda create -p venv python==3.12

conda activate venv/
```
### Step 2: Create a requirements.txt file 
```
conda install --file requirements.txt
```

### Step 3: Create a setup.py file 
```
This is to install the entire project as a package. Also to install packages mentioned in requirements.txt
```

### Step 4: Create the folder structure 
```
- src
    - components
    - pipelines
- notebooks
    - data
- templates
```
#### `components` includes data_ingestion, data_transformation, model trainer, and __init__.py. These modules are used in pipeline. 

#### `pipeline` includes training_pipeline, prediction_pipeline and __init__.py. These contain actual pipelines created with help of components.

#### `notebooks` includes EDA.ipynb and data folder that contains locally saved copy of data.

#### `templates` contains all the html template for the app.

### Step 5: Create other utility files
```
...
- src
    ...
    - exception.py
    - logger.py
    - utils.py
```
#### `exception.py` defines a custom execption class that is used throughout the peogram to handle exceptions.

#### `logger.py` defines configurations for logging. 

#### `utils.py` contains random functions frequently used in different modules of the project.

### Step 6: Create `app.py` for the webapp to render html pages using flask