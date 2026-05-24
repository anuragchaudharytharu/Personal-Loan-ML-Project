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