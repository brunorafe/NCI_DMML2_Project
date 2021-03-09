# A Deep Learning Approach to Supply Chain Data

The mentioned code is a study of supply chain data using deep learning. The dataset selected is based on a product inventory of an anonymous company, and it has information about quantities of products, previous sales and forecast sales. The study is focused on applying an ANN model in a supply chain dataset to forecast when a product will be in backorder with different proportions of the dataset using downsampling and upsampling techniques. The results were evaluated by accuracy, precision, recall and F1 score metrics as well as ROC Curve and Confusion Matrix methods.

# Data Source

The dataset selected to conduct the study is the [Back Order Prediction](https://www.kaggle.com/adityanarayansinha/back-order-prediction-using-ann), which is basically based on product inventory data. The dataset has 23 features and approximately one million observations, which are products, in our case. The features are mostly divided by the following information:
- The lead time of the product's delivering process.
- Quantity of products in transit.
- The forecast sales for the next three, six and nine
months.
- The previous sales of the last nine months.
- The average performance of the last six and twelve
months.

The main task of the dataset is to predict if a product is
going to be on backorder based on quantity, forecast and sales
data.

# Preprocessing

The following data mining steps were conducted in order to make all data sources ready for model's application:

- **Inconsistency Check**: activities regarding the verification of inconsistencies in the raw data such as incomplete observations or bad format data. Check up of the correct type of data attribution is also covered in this section.
- **Data Cleaning**: activities regarding the cleaning of incomplete or corrupt format data.
- **Feature Engineering**: activities regarding the selection and creation of relevant attributes which contribute most to the predicted outputs.
- **Balancing**: activities regarding the definition of balance of the target output of data. Sampling techniques will also be presented in this step.
- **Shuffling**: activities related to the rearrangement of the data in a random way.
- **One-Hot Encoding**: activities regarding the conversion of categorical data into a computer-based format.
- **Feature Scaling**: activities regarding the normalisation or standardisation of inputs which will be used by machine learning models.

# Development Setup

The project was created based on Anaconda environment. The following libraries are necessary in order to run the codes:
- re
- numpy
- pandas
- seaborn
- matplotlib.pyplot
- tensorflow

All libraries are already installed in Anaconda environment default installation with exception of the tensorflow library, which requires [additional procedures](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/) for installation. For more information about how to install Anaconda environment, please refer to its [documentation](https://www.anaconda.com/products/individual).

# Usage

The codes are divided as follows:

| Code | Description|
|:-----------------------------------------:|:----------------------------------------------------------------------------------------------------------------:|
| Group_B-DMML2-Data_Preprocessing_v5.ipynb | The preprocessing steps necessary to implement the ANN model.|
| Group_B-DMML2-Data_Model_v5.ipynb         | The application of the ANN model as well as its evaluation using accuracy metric, ROC curve and confusion matrix.|

# Results

In supervised learning applications, it is essential to keep a balance among all output classes since the models would not be biased by one class or another. This issue is also known as the unbalanced problem, and unfortunately, it is common in several real-life applications.

The dataset shows an unbalanced class problem with the majority of negative classes, where the positive class represents only 0.85% of the dataset, which is typical since rarely a product will be in a backorder situation. Figure below illustrates the unbalanced dataset of the output feature (went_on_backorder):

|![](/Figures/backorder_balance_binary.png) |
|:-----------------------------------------:|
| Balance of the original dataset           | 

After trying to train ANN model using the original balance of the data, the results did not show any prediction capacity when predicting the positive class, obtaining a **precision of 0.0%** and a **recall of 0.0%** on the test data. It is believed that the model was able to learn only about the negative class, given the unbalanced issue.

To deal with this situation, a Data Sampling solution has been considered. The Data Sampling approach is used to modify training datasets in such a way to increase the proportion of the class with fewer observations by either downsampling the majority class or upsampling the minority one.
This work has considered three experiments of Data Sampling in order to compare them and determine a suitable method to teach the models about products going on backorder, as follows:

<table>
<thead>
  <tr>
    <th align="center" rowspan="2">Experiments</th>
    <th align="center" colspan="3">Proportion of Positive and Negative Classes</th>
  </tr>
  <tr>
    <td align="center">Name</td>
    <td align="center">Positive Classes</td>
    <td align="center">Negative Classes</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">1</td>
    <td align="center">Down-sampled 50/50</td>
    <td align="center">8.000 (50%)</td>
    <td align="center">8.000 (50%)</td>
  </tr>
  <tr>
    <td align="center">2</td>
    <td align="center">Down-sampled 05/95</td>
    <td align="center">8.000 (5%)</td>
    <td align="center">150.000 (95%)</td>
  </tr>
  <tr>
    <td align="center">3</td>
    <td align="center">Up-sampled</td>
    <td align="center">40.000 (50%)</td>
    <td align="center">40.000 (50%)</td>
  </tr>
  <tr>
    <td align="center">For all</td>
    <td align="center">Test For All</td>
    <td align="center">900 (0.85%)</td>
    <td align="center">100.000 (99.15%)</td>
  </tr>
</tbody>
</table>

Given the structure of the dataset studied on this work, it was decided to create a supervised classification model using an ANN model. 

For the final model, two hidden layers were used, both using the activation function Relu, which seemed to be appropriated, considering that most of the features present only positive values. The sizes of these layers are 15 for each one. They were adjusted through several iterations aiming better results. 

A final dense layer was used with a Sigmoid function to get the output for the binary classification of classes, given the probabilities that the network returns. The optimiser used was Adam, which is an enhancement of Stochastic Gradient Descendent (SGD) and a Binary Cross-entropy loss function, given the fact that this is a binary classification problem. 

The number of epochs was decided through iterations, where the history was observed on each of them, plots for different metrics relevant for this problem were used, and the best results were obtained at 150 epochs. Even with the changes proposed the number the loss function, the AUC value, precision, and recall did not improve. The table bellow shows a summary of the model used for all the datasets designed for the experiments:

<table>
<thead>
  <tr>
    <th align="center">Layer</th>
    <th align="center">Size</th>
    <th align="center">Activation Function</th>
    <th align="center">Optimizer</th>
    <th align="center">Loss Function</th>
    <th align="center">Epochs</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">Dense</td>
    <td align="center">15</td>
    <td align="center">Relu</td>
    <td align="center" rowspan="3">Adam</td>
    <td align="center" rowspan="3">Binary Crossentropy</td>
    <td align="center" rowspan="3">150</td>
  </tr>
  <tr>
    <td align="center">Dense</td>
    <td align="center">15</td>
    <td align="center">Relu</td>
  </tr>
  <tr>
    <td align="center">Dense</td>
    <td align="center">1</td>
    <td align="center">Sigmoid</td>
  </tr>
</tbody>
</table>

To evaluate the model and compare the three experiments designed, each one of the datasets were created considering a single test data with the balance defined for them. In addition to that, all the models trained with its experimental datasets were tested with a global test dataset, which has the original balance of classes.

Results have shown that the model trained with the Experiment 2, the **"Down-sampled 05/95%"** model, performed better, obtaining the lowest value for the loss function and a good accuracy when tested with its single **test dataset (0.949)** and with the **"test for all" (0.988)**. The other experiments obtained a relatively good accuracy only when they were tested in their single test dataset (0.788 and 0.813, respectively). However, when they tested with the original balance of the data they are not able to get good predictions, obtaining accuracy of 0.137 and 0.144, which means that training the models with balanced classes do not achieve a proper generalisation of the data.

![](/Figures/result_accuracy_loss.png)

Observing the ROC AUC values, the Up-sampled model has the best result with 0.882. However, testing on the "test for all" dataset it is the **"Down-sampled 05/95%"** model the one with the best results again, obtaining a ROC AUC value of 0.841, while models from Experiments 1 and 3 got 0.741 and 0.767 respectively. The figures below show the ROC curves for the three models, tested both in their single test dataset and in the test dataset with the original balance of classes.

![](/Figures/roc_curve_all_test.png)

![](/Figures/roc_curve_single_test.png)

The results of accuracy and loss metrics, as well as the ROC curve can give the false perception that the **"Down-sampled 05/95%"** model were accurate and the one with best results. However, the goal of the study was to create a model, which would be able to identify products in backorder situations (in our case, the positive classes). The Confusion Matrix helps us to see exactly which classes were predicted correctly. By looking at the Confusion Matrix results below, it is possible to see that the experiment using the up-sampling technique (**Up-Sampled Model**) was the best choice while trying to predict products in backorder situation:

![](/Figures/confusion_matrix_all.png)

![](/Figures/confusion_matrix_single.png)
