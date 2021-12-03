# Installation

- on mac, I needed to use python 3.6 to use pytorch.
- download relevant distribution here: https://www.python.org/downloads/

`virtualenv -p python3.6 .`

`source bin/activate`

`pip3 install -f requirements.txt`

- If you prefer Anaconda package management:

`conda env create -f environment.yaml`

`conda activate lyric_sim`


# Adding a dependency
`pip3 install <dependency>`

`pip3 freeze > requirements.txt`

- Anaconda install dependency 

`conda install <dependency>`

# .gitignore
- please add any platform specific files like .DS_Store or .vscode to the .gitignore

# Get the datasets

- I have both wget and wget2 on my machine. If you don't have those, use brew to install them, or just use curl.

`./scripts/download_datasets`

`./scripts/extract_datasets`

- If you need to add new files to the project, add them to urls.txt
- The files are downloaded into the data_files directory

# Notebooks

- if you commit changes to a notebook, please clear large cell outputs before commiting.

# Training a model
`python your_model --config <your_config>`
- configs should use the .yaml extension.
- don't pass the extension to the argument. i.e. --config default

