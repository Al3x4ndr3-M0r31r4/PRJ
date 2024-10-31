
# Setup

The first task to complete is to install Anaconda: https://docs.anaconda.com/anaconda/install/


## Create Environment

The next step is run the two scripts in the Anaconda prompt. Firstly, you need to read and change the setupXAI1 file. This file creates 
the anaconda environment in a specified location with the libraries present in the envXAI.yml file. It is important to follow the comments 
that are present in this file. Secondly, you need to run the setupXAI2, which reinstalls the correct version of SHAP.


# Project Structure

The management of the project was made using PyCharm.

The main packages are the following:
	- configs
	- datasets
	- logs

In addition to this packages there are also the packages that are used to store the trained models, those are:
	- darwin_models
	- braintumor_models
	
In this packages are stored the models trained with entire dataset, also there are dummy input simulating the patients, but those are
actually part of the original dataset (rows). Also there is a output directory where the classification of each patient, as well as the 
explanations for it, are stored.


In the "configs" package, there are 4 files:
	- configs.json
	- input_settings_brain_tumor.json
	- input_settings_darwin.json
	- visual_config.json

The configs.json file has the path to the "datasets" and "logs" packages.

The input_settings_brain_tumor.json file has the paths to save and diagnose for the brain tumor dataset, including where are the
models saved, what is the folder where the patients informations are to make the diagnosis and to which directory should 
save the outputs of the diagnosis.

The input_settings_darwin.json has the same informations but for the darwin dataset

The visual_config.json file has some visual configurations used on the project, like the color of the bars, colors of shap and the
default title size.

The "datasets" package has all the datasets that were used in this project, namely the brain tumor dataset, which is divided into 2
folders: train and test. Each of this folders have 3 folders that represent each class of this dataset.
The DARWIN dataset is a .csv file.

The "logs" package will be where the output of each file is saved, both performance metrics and extracted explanations.


## File Notation

The files can have several suffixes, being them as follows:
	- Baseline
	- GS
	- CV
	- Save_Models
	- Diagnosis

The "Baseline" notation means that the hyperparameters of the classifiers were the default ones or that were made few alterations to these.

The "GS" notation means that the file in question is meant to find the most adequate hyperparameters for the dataset using Grid-Search methods 
for each classifier.

The "CV" notation means that Cross-Validation is used and that each classifier will be run several times with different train/test partitions.
When "CV" notation mixes with one of the above like "CV_Baseline" or "CV_GS" it means that the Cross-Validation is being applied to the 
models that have the default parameters or to the models that have fine-tuned parameters given by the Grid-Search.

The "Save_Models" notation means that the file in questions is meant to save each model in disk using all the dataset to train them to simulate
the real context where the models are trained with all the data available and then used on new patients.

The "Diagnosis" notation means that the file in question is meant to provide diagnosis and its explanations to 1 or more patients present in a
certain folder.


## File Structure

Each file has 3 initial sections:
	- Introduction
	- Imports
	- Setup

The "Introduction" section that briefly states the purpose of the file.
The "Imports" section where all the necessaryimports are stated.
The "Setup" section is where all the "environment variables" are loaded to be used further in the file, to create or to save the output.

The "Baseline" and "GS" files are similar in structure, firstly they make a brief examination of the dataset. Then they train each model first 
showing the independant metrics and in the end of the file is presented a summary of the obtained metrics and explainations which are also 
saved in this section.
The "Save_Models" is similar to the previous, the only diffence is that no explanations are extracted, it is only displayed the obtained
metrics for the train partition, since all the dataset is used for training.

The "CV" files first created all the partitions that will be used to train and test the classifiers. Then each classifier is trained with those
partitions, and finally in the Summary section the metrics for each partion are shown as well as its mean and standard deviation. Excluding the
Brain Tumor dataset, explanations are also extracted in this phase.

The "Diagnosis" files first load the data of the patients to diagnose, then all the models are loaded. In the next step, the models are used
to make predictions along with its explanations. Finally all the outputs are saved in the specified output file.

Note: To use the "Diagnosis" files you will have to unzip the "resnet.7z" file in the package "braintumor_models" and the "DARWIN_SVM_SHAP.7z" 
in the "darwin_models" package. This files had to be compressed due to file size limits impositions.


## Utility Files

Some files were created to auxiliate the others:
	- SyntheticDataset.py
	- DatasetAnalysis.py
	- DatasetFromDisk.py
	- PerformanceAnalysis.py
	- GradCAM.py
	- Visualization.py
	
The "SyntheticDataset.py" is used to generate the synthetic dataset used for experiments and its parameters are explained is the file. It can
have two main ways of operation: generating a synthetic dataset from noise or from gaussian distributions.

The "DatasetAnalysis.py" is meant to provide insights on the dataset prior to the classification itself. it has two functions: one to show
the distribution of the present classes and the other to show the histogram of each feature of the provided dataset.

The "DatasetFromDisk.py" is used for the Brain Tumor dataset. When the domain of the data is image, it is often not possible to load all the 
dataset to memory, so this class is used to load the dataset from disk to memory in batches using keras and tensorflow. It also makes all the
preprocessing needed for the images, namely colormapping, rescaling and domain conversion from 12-bit to 8-bit.

The "PerformanceAnalysis.py" is used to evaluate the performance of the classifiers. This evaluation can assume the form of a confusion matrix,
a PR or ROC curve, or other metrics taken from the confusion matrix like accuracy and recall.

The "GradCAM.py" file is used to extract explanations from the CNN used for the Brain Tumor dataset, it highlights via heatmap the reagion of
the image that is the most important.

The "Visualization.py" is an auxiliary that facilitates greatly the visualization of the confusion matrices and of the extracted explanations,
making displaying them in agrid and/or saving them in disk.



# Acknowledgments

This research was supported by Instituto Politécnico de Lisboa (IPL) under Grant IPL/IDI&CA2024/ML4EP_ISEL.




