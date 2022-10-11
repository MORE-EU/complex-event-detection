(initial integrated version of the platform)
Complex-Event-Detection
=======================

  Complex-Event-Detection is a python library for implementing utilities which aim to fulfil a full Event Detection  workflow for analysing time series data. Complex-Event-Detection implements a series of training features, for the input time series, running variations of deviation detection algorithms in order to identify change points with a mixture of supervised and semi supervised methods so to achieve real time results , then behavior detection so to create two type of models, in order to find the most fitting model for the data at hand. Lastly a motif detection tool to extract motifs for summarized time series and for the deviation detection tool.  
  
Description
===========	
For the development process, several python libraries will be used. In particular: [STUMPY](https://stumpy.readthedocs.io/en/latest/), [MatrixProfile](https://matrixprofile.docs.matrixprofile.org/), [Pyscamp](https://pypi.org/project/pyscamp/) and [Scikit-Learn](https://scikit-learn.org/stable/) . The aforementioned libraries are implemented in [python3](https://www.python.org/download/releases/3.0/) (or provide python3 bindings to C++ code), thus the Event Detections module will adopt the same programming language. An installation of the [CUDA toolkit8](https://developer.nvidia.com/cuda-toolkit) is necessary for deploying the GPU-accelerated versions of the aforementioned libraries. 
The module consists of the following steps:

Complex Event Detection tools
=============================
  The execution of the project starts with the presentation of the tools. 

1. **Deviation Detection**

  In this section, we present a tool for detecting segments of a time series where the behavior of a given target variable deviates much from the usual. We assume that certain 
  segments of the time series are labelled, representing the usual behavior of the variable in question. In this tool the user can find the list of variants that took place to implement it.

  * Self-supervised changepoint detection: This refers to our method for detecting changepoints in one variable’s behavior as follows: given a set of segments of the input time series, our method decides for each one of those segments whether it contains one changepoint
  
  * Semi-supervised changepoint detection: This also refers to a method for detecting changepoints in one variable’s behavior, similar to the above. The main difference is that we assume minimal knowledge

  * Self-supervised modelling: This component refers to regression models that capture the target variable’s expected behavior
  * Semi-supervised modelling: This refers to the regression model used in the semi-supervised changepoint detection component that aims to capture the expected behavior of our target variable
  * Deviation detection:  Our models for the expected behavior aim to provide a tool for detecting periods where the target variable has a deviating behavior. This deviating behavior is typically slowly progressing, so it cannot be detected as a changepoint, since changepoints refer to rapid changes.  This component provides a toolkit for analysing historical data and detecting deviating periods in a time series which is completely known when our models are deployed.    
  * Real-time deviation detection: Using our models for the expected behavior of a target variable, we are also able to make real-time deviation detection. Real-time deviation detection refers to a scenario where new data points are received, as a stream.
  
  Link to the notebook:[Deviation Detection](https://github.com/MORE-EU/more-pattern-extraction/blob/main/notebooks/deviation_detection.ipynb)


2. **Behavior Detection**

 Handling the problem as an event detection task, introducing the requirement to be able to deploy the developed methods in a real-time setting. Namely, our methods need to effectively and efficiently/scalably predict in constantly incoming windows of multidimensional time series, produced by a large number of different turbines in parallel. In our setting, the multidimensional time series consists of a set of variables-measurements on the turbine. The desired event-behavior we aim to detect is the (absolute value of the) static angle between nacelle and wind direction, termed yaw misalignment angle. Given that, we consider two different approaches for modeling and solving the task.
	
1. Directly model: use the yaw misalignment angle as a dependent variable and train regression models that exploit the remaining time series variables as independent variables to predict the former. The regression models are either to be learned on a historical data and be deployed (tested) on the newly incoming data of the same turbine or  to be learned on a time series and deployed on different series. This comprises the most intuitively straightforward approach, allowing to experiment with less variants. 

2. Indirectly model: the yaw misalignment angle by training regression models for approximating the dependent variable and assigning them to different angles. The prediction on newly incoming data is then performed by aggregation of the assigned angles of individual regression models that better approximate the dependent variable on a time window of the new time series. This comprises a more elaborate modelling approach, that allows us to experiment with more variants, e.g., considering aggregation schemes. 


The steps of the Behavior detection tool are presented here.
* Feature Selection: The component which is responsible for the detection of the most important variables.
* Binning: This refers to partitioning the dataset into bins based on the values of an input variable and train specific models on each bins.
* Model Tuning: Tuning the hyper parameters for each model
* Direct Modelling: This component refers to methods modelling the variable that represents the behavior of the time-series at the given point, as the dependent variable. 
* Indirect Modelling: This component includes solutions that work by modelling the behavior of the time-series during (labelled) periods and then assigning labels depending on how well those models approximate the behavior of the newly incoming data. 
* Behavior detection. Refers to the detection of the behavior of newly incoming data based on the available labels and utilizing an aggregate of the predictions of the multiple models trained in different bins or regions of the training datasets. 

  Link to the notebook: [Behavior Detection](https://github.com/MORE-EU/more-pattern-extraction/blob/main/notebooks/semantic_segmentation.ipynb)

3. **Motif Discovery**

  This group of methods continues part of our work documented in [Pattern extraction methods](https://github.com/MORE-EU/more-pattern-extraction), by examining how the Matrix Profile suite of algorithms can be adapted or extended in order to specialize for handling the use cases. In particular, in the current we briefly report on:
  
* Our ongoing work on adapting the Matrix Profile computation process to be applicable directly on the summarized (modelled) time series representation. 
* Our initial work on exploring Annotation Vector spaces, with respect to the time series values they utilize, that are suitable for focusing the motif extraction process specifically towards soiling events. 
 
  Link to the notebook: [Motif Discovery](https://github.com/MORE-EU/more-pattern-extraction/blob/main/notebooks/changepoint_detection.ipynb)
  

Documentation
=============

Source code documentation is available from GitHub pages [Link](https://more-eu.github.io/more-pattern-extraction/)
