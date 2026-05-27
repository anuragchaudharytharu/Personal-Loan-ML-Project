## End to End ML Project Project

## ML systems are built

⚠️ Experiment → Modularize → Pipeline → Deploy


## Code Writing Workflows

1. Research in notebook
2. Update config.yaml
3. Update schema.yaml (if dataset/schema changes)
4. Update params.yaml (if model params needed)
5. Update entity
6. Update configuration manager
7. Update component
8. Update pipeline
9. Update main.py
10. Update app.py (only for inference/UI)
11. Run pipeline
12. Debug logs/artifacts


✅ 1. Research in notebook FIRST

    research/*.ipynb

This is where:

1. preprocessing is tested

2. models are compared

3. feature engineering is explored

⚠️ Notebook experimentation ALWAYS comes first


✅ 2. Update config.yaml

Add:

1. paths

2. directories

3. URLs

4. configurable settings

🚀 Example

    data_transformation:
    root_dir: artifacts/data_transformation


✅ 3. Update schema.yaml

ONLY if:

1. columns changed

2. target changed

3. datatypes changed

🚀 Example

    COLUMNS:
    Age: int64
    Income: int64

⚠️ Not every stage needs schema update


✅ 4. Update params.yaml

ONLY for:

1. model training

2. hyperparameters

3. tunable configs

🚀 Example

    RandomForest:
    n_estimators: 200

⚠️ Ingestion/validation usually don’t need params.yaml   


✅ 5. Update Entity

Create/update dataclass.

🚀 Example

    @dataclass
    class DataTransformationConfig:


✅ 6. Update Configuration Manager

Connect:

    yaml → entity

🚀 Example

    get_data_transformation_config()


✅ 7. Update Component

This is where:

    actual ML logic lives

Examples:

1. data transformation
2. model training
3. data validation
4. data ingestion
5. model evaluation


✅ 8. Update Pipeline

Pipeline wraps component execution.

🚀 Example

    DataTransformationTrainingPipeline


✅ 9. Update main.py

Only if:

1. new stage added

2. stage order changed

🚀 Example

    Transformation → Trainer

⚠️ You do NOT always modify main.py

Only when pipeline flow changes.


✅ 10. Update app.py

ONLY for:

1. prediction UI

2. Flask/FastAPI/Streamlit

⚠️ app.py is NOT part of training pipeline

It belongs to:

    inference/deployment layer


🚀 COMPLETE MENTAL MODEL

| File             | Role                  |
| ---------------- | --------------------- |
| research         | experimentation       |
| config.yaml      | settings              |
| schema.yaml      | dataset structure     |
| params.yaml      | model hyperparameters |
| entity           | typed config objects  |
| configuration.py | yaml loader           |
| component        | actual logic          |
| pipeline         | stage execution       |
| main.py          | orchestration         |
| app.py           | prediction UI/API     |



    Step 1: (optional) create .venv vitual environament

        python -m venv .venv
    
    Step 2: (if .venv created) activate

        .venv\Scripts\activate

    Step 3: Install Requirements packages
        
        pip install -r requirements.txt

    Step 4: (if "-e ." is not included inside requirements.txt file)

        pip install -e .   

    When you do this:

    pip internally:

        Reads setup.py

        Executes it

        Finds packages

        Creates installation metadata
        
        Creates .egg-info

    Step 5: (if needed) run template file

        python template.py 

    Step 6: run app file

        python app.py