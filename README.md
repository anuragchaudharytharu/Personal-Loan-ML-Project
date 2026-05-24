## End to End ML Project Project

## Workflows

1. update config.yaml
2. update schema.yaml
3. update params.yaml
4. update the entity 
5. update the configuration manager in src config
6. update the components
7. update the pipeline
8. update the main.py
9. update the app.py

    Step 1: (optional) create venv and activate venv

        .venv\Scripts\activate

    Step 2: pip install -r requirements.txt

    Step 3: pip install -e .   (if not included in requirements)

    When you do:
                pip install -e .

    pip internally:

        Reads setup.py

        Executes it

        Finds packages

        Creates installation metadata
        
        Creates .egg-info

    Step 4: run template.py (if needed)

    Step 5: run app.py