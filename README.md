# A Deep Learning Approach to Supply Chain Data

Abstract—This electronic document is a proposed study of supply chain data. It shows the motivation behind the selection of data as well as a critical view of related works using deep learning techniques in supply chain networks. The dataset selected is based on a product inventory of an anonymous company, and it has information about quantities of products, previous sales and forecast sales. The study is focused on applying an ANN model in a supply chain dataset to forecast when a product will be in backorder with different proportions of the dataset using downsampling and upsampling techniques. The results were evaluated by accuracy, precision, recall and F1 score metrics as well as ROC Curve and Confusion Matrix methods.

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

All libraries are already installed in Anaconda environment default installation. For more information about how to install Anaconda environment, please refer to its [documentation](https://www.anaconda.com/products/individual).

# Usage

# Results

![](/Figures/result_accuracy_loss.png)

![](/Figures/roc_curve_all_test.png)

![](/Figures/roc_curve_single_test.png)

![](/Figures/confusion_matrix_all.png)

![](/Figures/confusion_matrix_single.png)
