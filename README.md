
# NIPS-Reproducibility-Challenge

## Paper: Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models

### Prerequisites for Getting Started

* Code in the github repository.
* Data sources.
  * [ECG Data](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000)
  * [Wafer Data](http://www.timeseriesclassification.com/description.php?Dataset=Wafer)
  * [Traffic Data](https://github.com/laiguokun/multivariate-time-series-data)
  
* Related papers.
  * [Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models]()
  * [Soft-DTW: a Differentiable Loss Function for Time-Series](https://arxiv.org/pdf/1703.01541.pdf)
  * [Introducing the Temporal Distortion Index to perform a bidimensional analysis of renewable energy forecast](https://www.sciencedirect.com/science/article/pii/S0360544215014619)
  
### Content

#### Project folder

* In the models folder, you can find four machine learning architectures. (We mainly use conv_lstm, fnn, seq2seq these three models to do the reproduction work.)
* In the loss folder, there are custom loss function dilate loss and it's back propagation implementation.
* alpha_test and gamma_test folders contain our experiments on parameter \alpha and \gamma, and all the files can be executed on Jupyter notebook.
*
*
