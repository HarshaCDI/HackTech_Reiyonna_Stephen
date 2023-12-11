#!/bin/bash

# Download training data
wget -O DETRAC-train-data.zip http://detrac-db.rit.albany.edu/Data/DETRAC-train-data.zip
unzip -q DETRAC-train-data.zip -d data/

# Download training annotations
wget -O DETRAC-Train-Annotations-XML.zip http://detrac-db.rit.albany.edu/Data/DETRAC-Train-Annotations-XML.zip
unzip -q DETRAC-Train-Annotations-XML.zip -d data/

# Download test data
wget -O DETRAC-test-data.zip http://detrac-db.rit.albany.edu/Data/DETRAC-test-data.zip
mkdir -p data/DETRAC  # create the DETRAC directory if it doesn't exist
unzip -q DETRAC-test-data.zip -d data/

# Download test annotations
wget -O DETRAC-Test-Annotations-XML.zip http://detrac-db.rit.albany.edu/Data/DETRAC-Test-Annotations-XML.zip
unzip -q DETRAC-Test-Annotations-XML.zip -d data/

python preprocessing/train_yolo_coordinates.py > train_yolo_coordinates.py.log 2>&1
python preprocessing/test_yolo_coordinates.py > test_yolo_coordinates.log 2>&1

pip install ultralytics
