import streamlit as st

st.header('Important Definitions')


general_terms_tab, models_tab, model_metrics_tab = st.tabs(['ML Terms', 'Models', 'Model Metrics'])


#Begin General Terms Tab
with general_terms_tab:
    st.subheader('Machine Learning Terms')
    #Supervised learning
    with st.expander('What is Supervised Learning?'):
        st.markdown("Supervised learning is a type of machine learning that is trained on labeled data sets. Labeled data sets "
                    "provide the algorithim with the correct output value to allow for the model to correct itself over iterations. "
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

#Begin Models Tab
with models_tab:
    st.subheader('Classification Models')

    with st.expander('k-Nearest Neighbors'):
        st.markdown("k-Nearest Neighbors (k-NN) is an algorithim that classifies new samples according to the class of the "
                    "samples closest to them (i.e, their nearest neighbors). The number of samples that the algorithim takes "
                    "into account is defined by the parameter k. Suppose that of the k nearest neighbors, some neighbors have "
                    "different classes. In this case one of two approaches are typically used to decide the new sample's class.  \n")

        st.markdown("  \n1. **Simple Majority**: The new sample's class will be set to whichever class has the greatest proportion "
                    "of the k nearest neighbors.  "
                    "\n 2. **Distance Weighted Voting**: Typically called Weighted k-NN, this algorithim gives additional weight "
                    "to neighbors that are closer to the new sample and reduces the weight for neighbors that are further "
                    "away from the sample. This makes neighbors closest to the new sample have a greater impact on the sample's "
                    "final class.")

    with st.expander('Random Forest Classifier'):
        st.markdown("The Random Forest Classifier is an algorithim that uses a collection of decision "
                    "trees to classify new samples. This collection of decision trees is called a forest. Since each decision "
                    "tree is trained on a different portion of the data, each decision tree is likely different from the others. "
                    "After each decision tree has produced an output, a simple majority is taken to determine the "
                    "final output of the algorithim.")

    with st.expander('Support Vector Machine'):
        st.markdown('placeholder')


    st.subheader('Regression Models')

    with st.expander('Histogram-Based Gradient Regression Tree'):
        st.markdown('placeholder')

    with st.expander('Random Forest Regressor'):
        st.markdown('placeholder')

    with st.expander('Ridge Regressor'):
        st.markdown('placeholder')

#End Models Tab

with model_metrics_tab:
    #TODO: Come up with a persistent example for each definition like how each metric applies to a email spam model
    #TODO: Explain nuance of unbalanced datasets, or models in which FN or FP is more important than the other
    # Google ML Concepts is a great place to look
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

    #MSE
    with st.expander('What is Mean Squared Error (MSE)?'):
        st.text("placeholder")

    #RMSE
    with st.expander('What is Root Mean Squared Error (RMSE)?'):
        st.text("placeholder")

    #MAE
    with st.expander('What is Mean Absolute Error (MAE)?'):
        st.text("placeholder")

    #MAPE
    with st.expander('What is Mean Absolute Percentage Error (MAPE)?'):
        st.text("placeholder")

#End Model Metrics Tab

