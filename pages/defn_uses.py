import streamlit as st
import pandas as pd

st.header('Important Definitions')


general_terms_tab, preprocessing_tab, models_tab, model_metrics_tab = st.tabs(['ML Terms', 'Preprocessing', 'Models', 'Model Metrics'])


#Begin General Terms Tab
with general_terms_tab:
    st.subheader('Machine Learning Terms')
    #Supervised learning
    with st.expander('What is Supervised Learning?'):
        st.markdown("Supervised learning is a type of machine learning that is trained on labeled data sets. Labeled data sets "
                    "provide the algorithm with the correct output value to allow for the model to correct itself over iterations. "
                    "Supervised learning is used to predict the outcomes of unseen data and can be split into two main branches: "
                    "classification and regression.")
        st.markdown("**Classification**:  \nClassification is used to assign data into different categories or classes. A common "
                    "example of classification is predicting whether an email is spam or safe.")
        st.markdown("**Regression**:  \nRegression is used to predict continuous numerical values. For instance, a model that "
                    "predicts housing prices given the number of rooms, square footage, and zipcode is a form of regression model.")

    #Unsupervised Learning
    with st.expander('What is Unsupervised Learning?'):
        st.markdown("Unsupervised learning uses unlabeled data sets to train a model. This allows the model to search for hidden "
                    "relationships in the data. This typically requires larger training datasets than supervised models and results in "
                    "more complex models. Unsupervised learning has two common use cases: clustering and dimensionality reduction.")
        st.markdown("**Clustering**:  \nClustering seeks to group unlabeled data depending on their similarities. Clustering can "
                    "be used to group customers together depending on their shopping behavior. This could allow for better product "
                    "recommendations.")
        st.markdown("**Dimensionality Reduction**:  \nDimensionality reduction is used to decrease the number of explanatory variables "
                    "to allow for increased model performance and computational efficiency. This can be done by removing or combining "
                    "explanatory variables.")
        st.markdown("**Curse of Dimensionality**:  \nThe curse of dimensionality is a concept that pertains to the relationship "
                    "between the number of explanatory variables and generalizability. Generally, as more explanatory variables "
                    "are taken into account, the generalizability of the model will decrease given that the number of data points "
                    "remains constant.")

    #Overfitting
    with st.expander('What is Overfitting?'):
        st.markdown("Overfitting occurs when a model matches the training set so well that it is unable to correctly predict "
                    "unseen data. This is typically caused when the training set misrepresents real world data and/or if the model "
                    "is too complex.")

    #Underfitting
    with st.expander('What is Underfitting?'):
        st.markdown("Underfit models perform poorly on their training set and on unseen data. Underfitting is typically a symptom "
                    "of models that are too simple to recognize relationships and patters in data.")
#End General Terms Tab

#Begin Preprocessing Tab
with preprocessing_tab:

    st.subheader("Feature Scaling")
    with st.expander("What is Normalization?"):
        st.markdown("Normalization, or Min-Max Scaling, is a form of feature scaling that maps data to the "
                    "range [0, 1].  \n\n")
                    #TODO: Finish when to normalize section
                    # "**When to Normalize**:  \n"
                    # "Data should be normalized when the distribution of the data is unknown or does not follow a Gaussian distribution. "
                    # "Normalization is highly sensitive to outliers since the range of values is confined to [0,1]")

    with st.expander("What is Standardization?"):
        st.markdown("Standardization is a feature scaling technique that scales the data such that the mean "
                    "of the features is 0 and the standard deviation is 1. This makes the data have similar properties to "
                    "a Gaussian distribution, however, this does not make the data have the shape of a Gaussian distribution.  \n\n")
                    #TODO: finish when to standardize section
                    # "**When to Standardize**:  \n")



#Begin Models Tab
with models_tab:
    st.subheader('Classification Models')

    with st.expander('k-Nearest Neighbors'):
        st.markdown("k-Nearest Neighbors (k-NN) is an algorithm  that classifies new samples according to the class of the "
                    "samples closest to them (i.e, their nearest neighbors). The number of samples that the algorithm  takes "
                    "into account is defined by the parameter k. Suppose that of the k nearest neighbors, some neighbors have "
                    "different classes. In this case one of two approaches are typically used to decide the new sample's class.  \n")

        st.markdown("  \n1. **Simple Majority**: The new sample's class will be set to whichever class has the greatest proportion "
                    "of the k nearest neighbors.  "
                    "\n 2. **Distance Weighted Voting**: Typically called Weighted k-NN, this algorithm  gives additional weight "
                    "to neighbors that are closer to the new sample and reduces the weight for neighbors that are further "
                    "away from the sample. This makes neighbors closest to the new sample have a greater impact on the sample's "
                    "final class.")

    with st.expander('Random Forest Classifier'):
        st.markdown("The Random Forest Classifier is an algorithm  that uses a collection of decision "
                    "trees to classify new samples. This collection of decision trees is called a forest. Since each decision "
                    "tree is trained on a different portion of the data, each decision tree is likely different from the others. "
                    "After each decision tree has produced an output, a simple majority is taken to determine the "
                    "final output of the algorithm .")

    with st.expander('Support Vector Machine'):
        col1, col2 = st.columns(2, border = True)
        with col1:
            st.markdown("The Support Vector Machine algorithm seeks to classify samples by finding a hyperplane that best separates the data into two classes. "
                    "To maximize the hyperplanes margin, the algorithm takes the two closest data points to the hyperplane "
                    "(i.e the support vectors) and plots the hyperplane that maximizes the margin between the support vectors.")
        with col2:
            st.image('./assets/svm_diagram.png',
                 caption = "Two-dimensional example of a Support Vector Machine's hyperplane")


    st.subheader('Regression Models')

    with st.expander('Histogram-Based Gradient Boosting Regression Tree'):
        st.markdown("To understand the Histogram-Based Gradient Regression Tree (HistGBRT) it helps to understand how a "
                    "conventional Gradient Boosted Regression Tree (GBRT) works.  \n  \n"
                    "**GBRT**: Conventional gradient boosting uses an ensemble of decision trees to predict a continuous value. "
                    "These trees are combined one at a time with each subsequent tree aiming to fix the errors of the current ensemble of trees. "
                    "Gradient boosting starts with a prediction, which is usually the mean of the target variable. "
                    "Next, the algorithm calculates the residuals (i.e the difference between the predicted value and the actual value). "
                    "Then, the algorithm fits a new decision tree which aims to predict the newly calculated residuals. "
                    "Using the results of the new decision tree, the algorithm updates the predictions using the following formula:")
        st.latex(r"\text{new predictions} = \text{old prediction} + \text{learning rate} * \text{tree prediction} ")
        st.markdown("The algorithm repeats the residual calculations and prediction calculations for many iterations.  \n")
        st.markdown("**HistGBRT**: The HistGBRT operates similarly to the GBRT algorithm, however, instead of performing calculations "
                    "on each continuous feature value, the algorithm sorts them into bins, or buckets, of values. With these bins, "
                    "the algorithm can reduce the total number of calculations required to fit each decision tree. "
                    "This significantly reduces the computational cost and time to train the model, especially for large datasets.")

    with st.expander('Random Forest Regressor'):
        st.markdown("The Random Forest Regressor utilizes the same principles of the Random Forest Classifier model, however, "
                    "instead of taking a simple majority vote of its constituent trees, it averages the trees to get the predicted value.")

    with st.expander('Ridge Regressor'):
        st.markdown("Ridge regression is a form of linear regression that penalizes terms to prevent the model from being overfit to the training data.")

#End Models Tab

with model_metrics_tab:
    #TODO: Come up with a persistent example for each definition like how each metric applies to a email spam model
    #TODO: Explain nuance of unbalanced datasets, or models in which FN or FP is more important than the other
    
    st.subheader('Classification Metrics')

    #Accuracy
    with st.expander('What is Accuracy?'):
        st.markdown("Accuracy is the proportion of correct predictions, irrespective of the prediction's class.")
        st.latex(r"\text{Accuracy } = \frac{\text{\# of Correct Classifications}}{\text{Total \# of Classifications}}")

    #Recall
    with st.expander('What is Recall?'):
        st.markdown("Recall is the proportion of true positives that were correctly classified as positive. A higher recall "
                    "will result in a lower number of false negatives. False negatives are actual positives that were"
                    " incorrectly classified as negative, thus, the denominator of the following formula can be thought of as the sum"
                    " of all actual positives.")
        st.latex(r"\text{recall } = \frac{\text{true positives}}{\text{true positives} + \text{false negatives}}")
        st.markdown("Aliases: Sensitivity, True Positive Rate (TPR)")

    #Precision
    with st.expander('What is Precision?') :
        st.markdown("Precision is the proportion of correct predictions for the entire positive class."
                    "Higher precision values results in fewer false positives. Precision can be calculated with the following formula: ")
        st.latex(r"\text{precision } = \frac{\text{true positives}}{\text{true positives + false positives}}")

    #F-beta & F1
    with st.expander ('What is F-score? '):
        st.markdown("**F1**: The F1 score if the harmonic mean of precision and recall. If precision and recall have similar values,"
                    "then the F1 score will be close to their values. If precision and recall are dissimilar, "
                    "the F1 score will be closer to whatever metric is lower.")
        st.latex(r"\text{F1} = 2 * \frac{\text{precision} * \text{recall}}{\text{precision} + \text{recall}}")
        st.markdown(r"**F**$\beta$: F$\beta$ is the generalized form of F1. F$\beta$ allows for precision and recall to be weighted "
                    r"differently in the F-score calculation. When $\beta$ = 1, the score will be identical to an F1 score. "
                    r"$\beta$ scores higher than 1 will increase the weight for recall, while $\beta$ scores lower than 1 will "
                    r"increase the weight for precision.")
        st.latex(r"\text{F$\beta$} = (1 + \beta^2) * \frac{\text{precision} * \text{recall}}{(\beta^2 * \text{precision}) + \text{recall}}")

    st.subheader('Regression Metrics')

    #MSE and RMSE
    with st.expander('What is Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)?'):
        st.markdown("**MSE**")
        st.markdown("Mean Squared Error (MSE) is a regression metric that measures the accuracy of a model. MSE gives a larger penalty "
                    "to large errors which increases it's sensitivity to outliers. ")
        st.latex(r"\text{MSE} = \frac{1}{n} \sum^{n}_{i=1}(Y_i - \hat{Y}i)^2")
        st.markdown("In simpler terms, the MSE squares the difference between the actual value and the predicted value of a data point. "
                    "It then adds the squared differences for every data point. This summation is then divided by the total number of "
                    "data points, finally giving the MSE.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Pros:**\n\n- MSE is an easily understandable measure of a model's accuracy.\n- Easily calculated")
        with col2:
            st.markdown("**Cons:**\n\n"
                        "- Highly sensitive to outliers\n"
                        "- The units of MSE are squared units. Suppose the target is housing prices, then the units of MSE would be dollars².")

        st.markdown("**RMSE**")
        st.markdown("Root Mean Squared Error (RMSE) is simply the square root of the MSE value. They share many of the same "
                    "pros and cons. However, RMSE gives the units of the target variable instead of square units, thus making "
                    "it easier to interpret.")

    #MAE
    with st.expander('What is Mean Absolute Error (MAE)?'):
        st.markdown("Mean Absolute Error (MAE) measures the average difference between predicted and actual values. For example, "
                    "a MAE score of 5 indicates that, on average, each prediction is approximately 5 units away from its true value.")
        st.latex(r"\text{MAE} = \frac{1}{n} \sum^{n}_{i=1} |Y_i - \hat{Y}_i|")
        col1, col2  = st.columns(2)

        with col1:
            st.markdown("**Pros**\n\n"
                        "- Same units as target variable\n"
                        "- Less sensitive to outliers\n")
        with col2:
            st.markdown("**Cons**\n\n"
                        "- Since outliers are not penalized, MAE underreacts to them\n"
                        "- Absolute value function is non-differentiable at 0, so metric is not ideal for gradient-based models.")

    #MAPE
    with st.expander('What is Mean Absolute Percentage Error (MAPE)?'):
        st.markdown("The Mean Absolute Percentage Error measures how far off the model's predictions are as a percentage. "
                    "For instance, if the MAPE score was 10, then the model's predictions are off by 10% of the actual values, on average.")
        st.latex(r"\text{MAPE} = \frac{100\%}{n} \sum^{n}_{i=1} \left| \frac{Y_i - \hat{Y}_i}{Y_i} \right|")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Pros**\n\n"
                        "- Easy to interpret\n"
                        "- Standardized across datasets, error is a percentage instead of a unit-dependent value")
        with col2:
            st.markdown("**Cons**\n\n"
                        "- In the case that the actual value is zero, the formula will divide by zero, resulting in an undefined equation.\n"
                        "- MAPE cannot handle negative target values.\n"
                        "- Low actual values cause the MAPE calculation to exaggerate the impact of errors. ")


    with st.expander('Which Metric Should I Use?'):

        data = {
            "Metric": ["MSE", "RMSE", "MAE", "MAPE"],
            "Use Case": [
                "• Use when large errors should be penalized heavily\n• Use when model should be sensitive to outliers",
                "• Use when MSE the benefits of MSE are desired, but in the target variable's units",
                "• Use when equal weighting of all errors is desired\n• Use when you want errors to be displayed in the target variable's units",
                "• Use when you want errors to be displayed as a percentage\n• Use when you want to compare model performance across different units."
            ],
            "When to Avoid": [
                "• Avoid when outliers are not important",
                "• Avoid when the square root of MSE makes the value harder to interpret",
                "• Avoid when outliers are important",
                "• Avoid if target values include zero\n• Avoid if target values include very small values"
            ]
        }

        df = pd.DataFrame(data)

        # Convert to Markdown
        def df_to_markdown(df):
            header = "| " + " | ".join(df.columns) + " |"
            separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
            rows = [
                "| " + " | ".join(str(cell).replace("\n", "<br>") for cell in row) + " |"
                for row in df.to_numpy()
            ]
            return "\n".join([header, separator] + rows)

        # Display
        st.markdown(df_to_markdown(df), unsafe_allow_html=True)


#End Model Metrics Tab

