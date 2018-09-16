# Project: Capstone 
This project will use **Python** and **C#** programminng languages and use following libraries:
- sklearn.datasets, sklearn.model_selection, sklearn.metrics     
-  keras.utils, keras.preprocessing, keras.layers, keras.models, keras.callbacks
- numpy
- glob
- os
- cv2
- pandas
- tqdm 
- matplotlib.pyplot
The C# libraries used in this project are System, System.Diagnostics and System.IO.
## Data
Datasets are over 4GB size. Ilisted them  and their download adresses  below:
   - imgs.zip 4GB size images used in state farm distracted driver detection competition(file can be downloaded from [here](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data))
    - driver_imgs_list.csv.zip contains the csv file which will be used in this project(file can be downloaded from [here](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data))
    - files of the benchmark model 

# Project files
## Jupyter Notebooks
- Capstone_CNN.ipynb  : Contains all the CNN pre-processing and CNN design, test, and plotting codes.
- Capstone_Refinement.ipynb: Contains all the refined model preprocessing design ,test and plotting codes.
- Capstone_Transfer_Learning.ipynb :  Contains all the Transfer learning design, test and plotting codes.
## HTML Files
- Capstone_CNN : html version of Capstone_CNN.ipynb file.
- Capstone_Refinement : html version of Capstone_Refinement.ipynb file.
- Capstone_Transfer_Learning : html version of Capstone_Transfer_Learning.ipynb .
- Distracted Drivers : HTML version of plot on Exploratory Visualization section of the report.
## PDF File
- Capstone_Report : Overall report of the capstone project
## Capstone Folder
This folder is created as web application project folder in Visual Studio and Contains subfolders and files. Some of the important files and folders are : 
- Bin : html version of Capstone_CNN.ipynb file.
- Classifiers : Contains two python files CNN.py and Transfer.py for the classifiers.
    *  CNN.py  : Contains all of the CNN classifier code and another predict() function which predicts uploaded images using CNN model.
    *  Transfer.py : Contains all of the Transfer Learning classifier code and another predict() function which predicts uploaded images using Transfer Learning model.
- Uploaded: Users uploaded images will be saved to this folder.
- Default.aspx : Web page will be used to upload images and show the results. 



