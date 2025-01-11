# CODETECH.TASK-5
Name-SWARAJ TAWADE
Company-CODETECH IT SOLUTIONS
ID-CT0806GI
Domain-Phython Programming
Duration-12th December 2024 to12th January 2025
Mentor-Neela Santhosh Kumar
Overview of the Project

Project Overview: Spam Email Detection using Naive Bayes in Scikit-learn

Objective:

The primary objective of this project is to build a predictive model that can classify email messages as either spam or not spam (ham) based on the content of the email. The model helps in automating the process of spam detection using machine learning techniques, with a focus on achieving good accuracy and performance with real-world data.

Key Activities:

1. Data Preprocessing:
Loading Data: A dataset of emails is loaded, with labels indicating whether each email is spam or not.
Text Vectorization: The text in emails is converted into a numerical format using the CountVectorizer. This converts email content into a matrix of token counts (bag-of-words model), making it suitable for machine learning algorithms.

2. Splitting Data:
The dataset is split into training and test sets to allow the model to learn from part of the data and evaluate its performance on unseen data.

3. Model Training:
Naive Bayes Classifier: A Multinomial Naive Bayes classifier is used for the classification task. It is well-suited for text classification due to its probabilistic nature and the ability to handle high-dimensional data effectively.

4. Model Evaluation:
After training, the model is tested on the test set to evaluate its performance.
Metrics such as accuracy, confusion matrix, and classification report (precision, recall, F1-score) are used to measure the model's effectiveness.
Visualization: A confusion matrix is plotted to visualize the number of correct and incorrect predictions.

5. Prediction:
The model can then predict whether new unseen emails are spam or not, based on the learned patterns from the training data.
Technology Used:

Python: The project is implemented in Python due to its rich ecosystem of machine learning and data analysis libraries.

Libraries:

Pandas: For data manipulation and handling the dataset.

Scikit-learn:
For building and training the machine learning model, particularly using the Naive Bayes classifier.
For splitting the dataset into training and testing sets.
For generating evaluation metrics like accuracy, confusion matrix, and classification reports.
CountVectorizer: A tool in Scikit-learn used to convert text data into a numerical format (bag-of-words) suitable for machine learning models.
Matplotlib & Seaborn: Used for visualizing the confusion matrix and making the evaluation results easier to understand.
