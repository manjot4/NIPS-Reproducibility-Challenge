# NIPS-Reproducibility-Challenge
Shape and Time Distortion Loss forTraining Deep Time Series Forecasting Models



1  Prerequisites for Getting Started
•Code in the github repository.
•Data sources.
–ECG Data
–Wafer Data
–Traffic Data
•Related papers.
–  Shape  and  Time  Distortion  Loss  for  Training  Deep  TimeSeries Forecasting Models
–  Soft-DTW: a Differentiable Loss Function for Time-Series
–  Introducing  the  Temporal  Distortion  Index  to  perform  abidimensional analysis of renewable energy forecast
2  Content
2.1  Project folder
•In  the  models  folder,  you  can  find  four  machine  learning  architectures.(We  mainly  useconvlstm,fnn,seq2seqthese  three  models  to  do  thereproduction work.)
•In the loss folder, there are custom loss functiondilate lossand it’s backpropagation implementation.
•alphatestandgammatestfolders contain our experiments on parameterαandγ, and all the files can be executed on Jupyter notebook.
•datafolder includes all the data loaders which would be used when testingon different dataset.1
•difftestfolder  has  12  Jupyter  notebooks  which  consists  of  4  differentdataset runing on 3 models.•runoncnnlstmmodel.py,runonfcnnmodel.pyandrunonseq2seqmodel.pyare three python files that could be directly run on Pycharm.  These ex-periments are only used synthetic data.  If you want to try on differentdataset, you need to download other dataset and put data into correspond-ing dataloader
