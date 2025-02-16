The OpenML Data Cleaner tool consists of a few important files

Before running the code, make sure to install the packages specified in the requirements.txt file

The app works for Python 3.9.

The dashapp.py file contains the app itself with all the functionalities of the OpenML Data Cleaner.
To run the OpenML Data Cleaner, simply run this file.
* To run the file, you first have to add an OpenAI API key to the create_correct_cryp_section function in the create_section.py file, for the Cryptic Attribute Names correction to work as intended.
* The most optimal view of the tool currently requires you to zoom out to 67% (this will be fixed soon), otherwise the text and sections are too big.
* The app right now requires a user to upload a dataset as a .csv file. This will be changed soon.
* To see the error detections and corrections for all errors, it is advised to use the synthetic_data.csv file in the data folder of this repository

This dashapp.py file makes use of:
* the error_detection.py file, which contains functions for error detection
* the error_correctinon.py file, which contains functions for error correction
* the create_sections.py file, which is used to create certain sections within the tool

The pFAHES folder contains detection methods for the disguised missing values based on Qahtan et al. (2018)'s paper
* These functions are used in the detect_dmv() functinon within the error_detection.py file

The lookups folder is used to store some look-up files for the CrypticIdentifier() function as created by Zhang et al. (2023)

The cached_files folder is used to store files during the running of the dash app

The assets folder contains a .css file for the styling of some parts of the tool

The data folder contains datasets and other files that were used for the experiment notebooks

The experiment notebooks contain the procedure and results of the experiments (data quality and cryptic column name query optimizing)

