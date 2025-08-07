from copy import deepcopy

import shap
import streamlit as st
import pandas as pd
import numpy as np
from shap import TreeExplainer, LinearExplainer, KernelExplainer
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay, \
    mean_absolute_percentage_error, mean_absolute_error, root_mean_squared_error, mean_squared_error, make_scorer, \
    precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils.validation import check_is_fitted
import seaborn as sns

st.title('Supervised Learning')



#Checks to ensure that data is in the cache
if 'data_file_data' not in st.session_state:
    st.warning('Data not uploaded, models cannot be trained.')
    st.stop()
else:
    data = st.session_state['data_file_data']

model_tab, pred_tab = st.tabs(['Model', 'Predictions'])



# Begin Model Training Code --------------------------------------------------------------------------------------------

if 'data_file_data' in st.session_state:

    if 'model_comparison_history' not in st.session_state:
        st.session_state['model_comparison_history'] = []


    with model_tab:
        col1, col2 = st.columns(spec=2, gap='small', vertical_alignment='top')
        with col1:

            options_sup = st.selectbox(label='Select Prediction Type',
                                       options=['Classification',
                                                'Regression'])

            st.divider()

            data_options = {'Raw' : ('X_train', 'X_test')}

            if 'sup_encoder' in st.session_state:
                data_options['Encoded'] = ('X_train_encoded', 'X_test_encoded')

            if 'sup_raw_scaler' in st.session_state:
                data_options['Scaled'] = ('X_train_scaled', 'X_test_scaled')

            if 'X_train_encode_scaled' in st.session_state and 'sup_raw_scaler' in st.session_state:
                data_options['Encoded & Scaled'] = ('X_train_encode_scaled', 'X_test_encode_scaled')


            else:
                print('No encoder detected.')

            if all(key in st.session_state for key in ['X_train_red', 'X_test_red']):
                data_options['PCA Reduced'] = ('X_train_red', 'X_test_red')
            else:
                print("No DR Data")


            data_form = st.radio(label = 'Select Form of Data',
                                 options = data_options.keys(),
                                 horizontal = True
            )

            y_train = st.session_state['y_train']
            y_test = st.session_state['y_test']

            X_train_key, X_test_key = data_options[data_form]

            X_train = st.session_state[X_train_key]
            X_test = st.session_state[X_test_key]

            if data_form == 'Raw':
                if not X_train.select_dtypes(exclude='number').empty or not X_test.select_dtypes(exclude='number').empty:
                    st.warning("Non-numerical features will be dropped when handling raw data")

                X_train = X_train.select_dtypes(include='number')
                X_test = X_test.select_dtypes(include='number')



            #Begin Regression Code -----------------------------------------------------------------------------------------

            # Begin HistGradBoost --------------------------------------------------------------------------------------
            def get_hgbrt_model():
                """ creates Histgram Gradient Boosting Regressor Tree (HGBRT) model with user-defined parameters

                    Takes user input for the following model hyperparameters: loss function, quantile (if applicable),
                    maximum number of leaves for each tree, and the maximum number of trees and returns a HGBRT model
                    with the user specified parameters

                :return: A HGBRT model with user-defined parameters
                """
                loss_map = { 'Squared Error' : 'squared_error',
                             'Absolute Error' : 'absolute_error',
                             'Poisson' : 'poisson',
                             'Quantile' : 'quantile'}

                if np.all(y_train > 0):
                    loss_map['Gamma'] = 'gamma'


                loss_choice = st.selectbox(label='Choose Loss Function',
                                             options=loss_map.keys(),
                                             index=0)



                loss_selection = loss_map[loss_choice]

                if loss_selection == 'quantile':
                    quantile_value = st.number_input(label='Enter Quantile Value',
                                                     min_value=0.01,
                                                     max_value=1.0,
                                                     step=0.01,
                                                     format='%.2f')

                else:
                    quantile_value = None

                learn_rate = st.number_input(label='Enter Learning Rate',
                                             min_value=0.0,
                                             max_value=1.0,
                                             step=0.01,
                                             value=0.1,
                                             format=
                                             '%.2f')

                max_num_iter = st.number_input(label='Enter Maximum Number of Iterations',
                                               min_value=1,
                                               step=1,
                                               value=100)

                max_leaf = st.number_input(label='Enter Maximum Number of Leaves for Each Tree',
                                           min_value=2,
                                           value=31,
                                           step=1)

                return HistGradientBoostingRegressor(loss=loss_selection,
                                                     quantile=quantile_value,
                                                     learning_rate=learn_rate,
                                                     max_iter=max_num_iter,
                                                     max_leaf_nodes=max_leaf
                                                     )
            # End HistGradBoost Code -----------------------------------------------------------------------------------

            # Begin Random Forest Regressor Code -- --------------------------------------------------------------------
            def get_random_forest_reg_model():
                """ creates Random Forest Regressor model with user-defined parameters

                    Takes user input for the following model hyperparameters: number of estimators, criterion, minimum
                    number of samples for a split, and tree depth (if applicable) and returns a random forest regressor
                    model with the user-specified hyperparameters

                   :return: A Random Forest model with user-defined parameters
                """
                num_estimators = st.number_input(label='Enter the number of estimators.',
                                                 min_value=1,
                                                 step=1,
                                                 value=100)

                criterion_choice = st.selectbox(label='Select a criterion',
                                                  options=['Squared Error', 'Absolute Error', 'Friedman MSE',
                                                           'Poisson'])
                criterion_map = {'Squared Error' : 'squared_error',
                                 'Absolute Error' : 'absoute_error',
                                 'Friedman MSE' : 'friedman_mse',
                                 'Poisson' : 'poisson'}

                criterion_selection = criterion_map[criterion_choice]

                num_min_samples_split = st.number_input(
                    "Enter the minimum number of samples required to split an internal node",
                    min_value=2,
                    step=1,
                    value=2)

                enable_tree_depth = st.checkbox('Enable tree depth parameter',
                                                value=False)

                if enable_tree_depth:
                    tree_depth = st.number_input('Enter the maximum depth of each tree.',
                                                 min_value=1,
                                                 step=1)
                else:
                    tree_depth = None

                return RandomForestRegressor(n_estimators=num_estimators,
                                             criterion=criterion_selection,
                                             max_depth=tree_depth,
                                             min_samples_split=num_min_samples_split
                                             )
            # End Random Forest Regressor Code  ------------------------------------------------------------------------

            # Begin Ridge Code -----------------------------------------------------------------------------------------
            def get_ridge_model():
                """ Creates Ridge Regressor model with user-defined parameters

                    Takes user input for the following model hyperparameters: alpha, and returns a Ridge model with the
                    user specified parameters

                   :return: A Ridge Regressor model with user-defined parameters
                """
                alpha_value = st.number_input(label='Input Alpha Value',
                                              min_value=0.0,
                                              value=1.0,
                                              step=0.01,
                                              format='%.2f')

                return  Ridge(alpha=alpha_value)
            # End Ridge Code -------------------------------------------------------------------------------------------

            #Begin SVR Code -------------------------------------------------------------------------------------------
            def get_svr_model():
                """ Creates Support Vector Regressor (SVR) model with user-defined parameters

                    Takes user input for the following model hyperparameters: C value, kernel type, degree (if applicable),
                    and epsilon value and returns an SVR model with the user specified parameters


                     :return: An SVR model with user-defined parameters
                  """
                # Selecting C value
                c_value = st.number_input('Input C Value',
                                          min_value=0.0,
                                          value=1.0,
                                          step=0.01,
                                          format="%.2f")

                # Selecting Kernel
                kernel_choice = st.selectbox('Select a Kernel',
                                             ('Linear', 'Polynomial', 'Radial Basis Function', 'Sigmoid'),
                                             index=0)
                kernel_map = {'Linear': 'linear',
                              'Polynomial': 'poly',
                              'Radial Basis Function': 'rbf',
                              'Sigmoid' : 'sigmoid'
                              }

                kernel_selection = kernel_map[kernel_choice]
                # Sets degree to a default value in case kernel_type isn't polynomial and thus degree isn't declared
                if kernel_selection == 'poly':
                    st.number_input('Input degree',
                                    value=3,
                                    step=1,
                                    min_value=0)
                degree = 3

                epsilon_value = st.number_input('Input epsilon value',
                                                value=0.1,
                                                step=0.01,
                                                min_value=0.01)
                return svm.SVR(C=c_value,
                               kernel=kernel_selection,
                               degree=degree,
                               epsilon=epsilon_value)
            #End SVR Code ----------------------------------------------------------------------------------------------


            if options_sup == "Regression":
                if not st.session_state['target_is_number']:
                    st.error("Target variable is not a number. Regression cannot be used.")
                    st.stop()

                model_display_names = { 'Histogram Gradient Boosted Regression Tree' : get_hgbrt_model,
                                        'Random Forest Regressor' : get_random_forest_reg_model,
                                        'Ridge Regressor' :get_ridge_model,
                                        'Support Vector Regressor': get_svr_model
                }

                model_choice = st.selectbox('Choose Regression Algorithm', list(model_display_names.keys()))
                selected_model = model_display_names[model_choice]()



            # End Regression Code ------------------------------------------------------------------------------------------

            # Begin Classification Code ------------------------------------------------------------------------------------

        # Begin SVM Code ------------------------------------------------------------------------
            def get_svm_model():
                """ Creates Support Vector Machine (SVM) model with user-defined parameters

                    Takes user input for the following model hyperparameters: C value, kernel type, and degree (if applicable),
                    and returns an SVM model with the user specified parameters


                     :return: An SVM model with user-defined parameters
                  """
                # Selecting C value
                c_value = st.number_input('Input C Value',
                                          min_value=0.0,
                                          value=1.0,
                                          step=0.01,
                                          format="%.2f")

                # Selecting Kernel
                kernel_choice = st.selectbox('Select a Kernel',
                                           ('Linear', 'Polynomial', 'Radial Basis Function'),
                                           index=0)
                kernel_map = {'Linear' : 'linear',
                              'Polynomial' : 'poly',
                              'Radial Basis Function' : 'rbf'}

                kernel_selection = kernel_map[kernel_choice]

                if kernel_selection == 'poly':
                    degree= st.number_input('Input Degree',
                                            min_value= 0,
                                            value=3,
                                            step=1)
                # Sets degree to a default value in case kernel_type isn't polynomial and thus degree isn't declared
                degree = 3

                return svm.SVC(C=c_value,
                               kernel=kernel_selection,
                               degree=degree)
            # End SVM Code ---------------------------------------------------------------------------------------------

            # Begin k-NN Code ------------------------------------------------------------------------------------------
            def get_knn_model():
                """ Creates k-Nearest Neighbors (k-NN) model with user-defined parameters

                    Takes user input for the following model hyperparameters: k Value and returns an SVM model with the
                    user specified parameters

                     :return: An k-NN model with user-defined parameters
                """
                # Selecting k-Value
                k_value = st.number_input('Input K Value.',
                                         min_value=1,
                                         value=1)
                return KNeighborsClassifier(n_neighbors=k_value)
            # End k-NN Code --------------------------------------------------------------------------------------------

            # Begin Random Forest Classifier Code ----------------------------------------------------------------------
            def get_random_forest_class_model():
                """ Creates Random Forest Classifier (RFC) model with user-defined parameters

                    Takes user input for the following model hyperparameters: number of estimators, criterion, minimum
                    number of samples for a split, and tree depth (if applicable) and returns an SVM model with the
                    user specified parameters


                     :return: An RFC model with user-defined parameters
                  """
                #Selecting number of estimators
                num_estimators = st.number_input('Enter the Number of Estimators.',
                                                 min_value=1,
                                                 step=1,
                                                 value=100)
                #Selecting Criterion
                criterion_choice = st.selectbox('Select a Criterion',
                                                  ('Gini', 'Entropy', 'Log Loss'))
                criterion_map = {'Gini' : 'gini',
                                 'Entropy' : 'entropy',
                                 'Log Loss' : 'log_loss'}

                criterion_selection = criterion_map[criterion_choice]

                #Selecting minimum numbers of samples for a split
                num_min_samples_split = st.number_input(
                    "Enter the Minimum Number of Samples Required to Split an Internal Node",
                    min_value=2,
                    step=1,
                    value=2)

                #Enabling tree depth parameter
                enable_tree_depth = st.checkbox('Enable Tree Depth',
                                                value=False)

                #Selecting tree depth and creating model
                if enable_tree_depth:
                    tree_depth = st.number_input('Enter the Maximum Depth of Each Tree.',
                                                 min_value=1,
                                                 step=1)
                else:
                    tree_depth = None
                return RandomForestClassifier(n_estimators=num_estimators,
                                              criterion=criterion_selection,
                                              max_depth=tree_depth,
                                              min_samples_split=num_min_samples_split
                                              )


                # End Random Forest Classifier Code -------------------------------------------------------------------


                # Begin Logistic Regressor Code -----------------------------------------------------------------------
            #End Random Forest Classifier Code -------------------------------------------------------------------------

            #Begin Logistic Regression Code ----------------------------------------------------------------------------
            def get_logistic_regression_model():
                """ Creates Logistic Regression model with user-defined parameters

                    Takes user input for the following model hyperparameters: penalty, solver, C value (if applicable)
                    and returns a Logistic regression model with the user specified parameters


                    :return: A Logistic Regression model with user-defined parameters
                    """

                penalty_map = {'None' : None,
                                'L2' : 'l2',
                                'L1' : 'l1',
                                'Elastic Net' : 'elasticnet'
                              }

                penalty_choice = st.selectbox('Choose a regularization method', options=list(penalty_map.keys()))

                penalty_selection = penalty_map[penalty_choice]

                solver_options = {
                    'l1' : ['liblinear', 'saga'],
                    'l2' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                    'elasticnet' : ['saga'],
                    None : ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
                }

                valid_solvers = solver_options[penalty_selection]
                solver_selection = st.selectbox('Solver', options=valid_solvers)

                if penalty_selection != 'none':
                    c_value = st.number_input('Input C Value',
                                              value=1.0,
                                              step=0.01,
                                              min_value=0.01,
                                              format='%.2f')
                else:
                    c_value = None


                l1_ratio = st.slider("ElasticNet Mixing", 0.0, 1.0,
                                     0.5) if penalty_selection == 'elasticnet' else None


                params = {
                    'penalty' : penalty_selection,
                    'solver' : solver_selection
                }

                if c_value is not None:
                    params['C'] = c_value

                if penalty_selection == 'elasticnet':
                    params['l1_ratio'] = l1_ratio
                try:
                    return LogisticRegression(**params)
                except ValueError as e:
                    st.error(f"Invalid solver/penalty/multiclass combo: {e}")
                    st.stop()
            #End Logistic Regression Code ------------------------------------------------------------------------------

            if options_sup == 'Classification':
                model_display_names = {
                    'Support Vector Machine (SVM)' : get_svm_model,
                    'k-Nearest Neighbors (k-NN)' : get_knn_model,
                    'Random Forest Classifier' : get_random_forest_class_model,
                    'Logistic Regression' : get_logistic_regression_model
                }



                model_choice = st.selectbox('Choose Classification Algorithm', list(model_display_names.keys()))
                selected_model = model_display_names[model_choice]()


                    # End Classification Code --------------------------------------------------------------------------
            st.session_state['unfitted_model'] = selected_model

            # End Model Training Code ------------------------------------------------------------------------------------------

            # Begin Model Metrics Code -----------------------------------------------------------------------------------------
            with col2:
                st.subheader("Model Performance Metrics")

                if options_sup == "Classification":
                    label_encoder = LabelEncoder()
                    try:
                        y_train = label_encoder.fit_transform(y_train)
                        y_test = label_encoder.transform(y_test)
                    except ValueError as e:
                        st.error("Label encoding failed, check prediction type.")
                        if st.button('View Exception'):
                            st.error(e)
                        st.stop()

                    # Store the encoder in session_state if needed for later inverse_transform
                    st.session_state["label_encoder"] = label_encoder

                try:
                    y_pred = selected_model.predict(X_test)
                except NotFittedError:
                    try:
                        selected_model.fit(X_train, y_train)
                        y_pred = selected_model.predict(X_test)

                    except ValueError as e:
                        st.write(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
                        st.write(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
                        st.subheader("X train dataframe")
                        st.dataframe(X_train)
                        st.subheader("X test dataframe")
                        st.dataframe(X_test)
                        st.error(f"Model fitting failed: {e}")
                        st.stop()

                except ValueError as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()


                if st.session_state.get("target_is_number", True):
                    metrics = {
                            'MSE' : mean_squared_error(y_test, y_pred),
                            'RMSE' : root_mean_squared_error(y_test, y_pred),
                            'MAE' : mean_absolute_error(y_test, y_pred),
                            'MAPE' : mean_absolute_percentage_error(y_test, y_pred)
                    }

                    display_cols = list(metrics.keys())
                    is_regression = True
                else:
                    metrics = {
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'F1': f1_score(y_test, y_pred, average='weighted'),
                        'Precision' : precision_score(y_test, y_pred, average='weighted'),
                        'Recall' : recall_score(y_test,y_pred, average='weighted')
                    }

                    display_cols = list(metrics.keys())
                    is_regression = False

                st.session_state['model_comparison_history'].append({
                    "Model": selected_model.__class__.__name__,
                    "is_regression" : is_regression,
                    "Metrics": metrics
                })

                #Begin Model Comparison History Code ------------------------------------------------------------------
                with st.expander('Model Comparison History'):
                    history_df = pd.DataFrame(st.session_state['model_comparison_history'])

                    if not history_df.empty and 'Metrics' in history_df.columns and history_df[
                        'Metrics'].notnull().all():
                        metrics_df = pd.json_normalize(history_df['Metrics'])
                        combined_df = pd.concat([history_df.drop(columns=['Metrics']), metrics_df], axis=1)

                        # Define metric groups
                        regression_metrics = ['MSE', 'RMSE', 'MAE', 'MAPE']
                        classification_metrics = ['Accuracy', 'F1', 'Precision', 'Recall']

                        # Filter rows by task type using options_sup
                        if options_sup == "Regression":
                            filtered_df = combined_df[combined_df['is_regression'] == True]
                            metric_cols = [col for col in regression_metrics if col in filtered_df.columns]
                        elif options_sup == "Classification":
                            filtered_df = combined_df[combined_df['is_regression'] == False]
                            metric_cols = [col for col in classification_metrics if col in filtered_df.columns]
                        else:
                            st.warning("Unknown task type.")
                            st.stop()

                        # Final display columns
                        display_cols = ['Model'] + metric_cols

                        # Show final filtered table
                        st.dataframe(filtered_df[display_cols])
                    else:
                        st.info("No models have been evaluated yet.")
                #End Model Comparison History Code ---------------------------------------------------------------------

                # Begin Cross Validation Code --------------------------------------------------------------------------
                with st.expander('Cross Validation'):
                    if 'cv_confirmation' not in st.session_state:
                        st.session_state['cv_confirmation'] = False

                    if 'cv_ran' not in st.session_state:
                        st.session_state['cv_ran'] = False

                    if not st.session_state['cv_ran'] and not st.session_state['cv_confirmation']:
                        st.warning("Cross validation can be computationally expensive. The number of folds has "
                                    "been limited to 10.")
                        if st.button("I understand"):
                            st.session_state['cv_confirmation'] = True
                            st.rerun()

                    else:
                        cv_folds = st.number_input('Enter number of cross validation folds',
                                                   min_value = 2,
                                                   max_value = 10,
                                                   step = 1)

                        if st.button('Begin Cross Validation'):
                            if options_sup == 'Classification':
                                scorers = {
                                            'accuracy': 'accuracy',
                                            'f1': make_scorer(f1_score, average='macro', zero_division=0),
                                            'precision': make_scorer(precision_score, average='macro', zero_division=0),
                                            'recall': make_scorer(recall_score, average='macro', zero_division=0),
                                        }
                                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

                            elif options_sup == 'Regression':
                                scorers = {'MAE' :  'neg_mean_absolute_error',
                                           'MSE' : 'neg_mean_squared_error',
                                           'MAPE' : 'neg_mean_absolute_percentage_error'
                                            }
                                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)



                            cv_scores = cross_validate(st.session_state['unfitted_model'],
                                                        X_train, y_train,
                                                        cv=cv,
                                                        n_jobs=-1,
                                                        scoring= scorers)
                            score_keys = [key for key in cv_scores if key.startswith('test_')]

                            # Create summary (mean ± std)
                            summary = {
                                key.replace('test_', ''): f"{cv_scores[key].mean():.3f} ± {cv_scores[key].std():.3f}"
                                for key in score_keys
                            }

                            # Convert to DataFrame for display
                            summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['Score'])

                            st.subheader('Cross-Validation Scores')
                            st.dataframe(summary_df)
                            fold_df = pd.DataFrame(
                                {k.replace('test_', ''): v for k, v in cv_scores.items() if k.startswith('test_')})

                            # Display table with fold-level results
                            st.subheader("Per-Fold Cross-Validation Scores")
                            st.dataframe(fold_df)

                            # Plot bar chart using matplotlib/seaborn
                            if options_sup == 'Classification':
                                st.subheader("Cross-Validation Scores per Fold")
                                fig, ax = plt.subplots(figsize=(8, 4))

                                # Convert to long format for easy plotting
                                df_long = fold_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
                                sns.barplot(data=df_long, x='index', y='Score', hue='Metric', ax=ax)

                                ax.set_xlabel('Fold')
                                ax.set_ylabel('Score')
                                ax.set_title('Cross-Validation Scores per Fold')
                                st.pyplot(fig)
                #End Cross Validation Code ---------------------------------------------------------------------------

                #Begin Classification Metrics ------------------------------------------------------------------------------
                if options_sup == 'Classification':
                    try:
                        check_is_fitted(selected_model)
                    except NotFittedError as e:
                        st.warning(f'Model has not been fitted to data. Metrics cannot be shown. {e}')


                    # Begin Classification Report Code--------------------------------------------------------------------------
                    with st.expander('Classification Report'):
                        class_report = classification_report(y_test, y_pred, output_dict=True)
                        class_report = pd.DataFrame(class_report).transpose()
                        st.table(class_report)
                    #End Classification Report Code ----------------------------------------------------------------------------

                    #Begin Confusion Matrix Code ----------------- -------------------------------------------------------------
                    with st.expander('Confusion Matrix'):
                        conf_mat = confusion_matrix(y_test, y_pred)

                        # Define custom labels
                        annot_labels = np.array([[str(cell) for cell in row] for row in conf_mat])

                        # Reset any existing figures
                        plt.close('all')

                        # Create themed figure
                        fig, ax = plt.subplots(figsize=(6, 4))
                        fig.set_facecolor('#0e1117')
                        ax.set_facecolor('#0e1117')


                        sns.heatmap(conf_mat,
                                    annot=annot_labels,
                                    fmt='',
                                    cmap='Blues',
                                    cbar=True,
                                    linewidths=0.5,
                                    linecolor='#0e1117',
                                    ax=ax)

                        # Style text and axes
                        ax.set_title('Confusion Matrix', color='white')
                        ax.set_xlabel('Predicted', color='white')
                        ax.set_ylabel('Actual', color='white')
                        ax.tick_params(colors='white', labelsize=10)

                        # Style colorbar ticks
                        cbar = ax.collections[0].colorbar
                        cbar.ax.yaxis.set_tick_params(color='white')
                        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

                        fig.tight_layout()
                        st.pyplot(fig)
                    # End Confusion Matrix Code---------------------------------------------------------------------------------

                # End Classification Metrics Code ------------------------------------------------------------------------------

                # Begin Regression Metrics Code --------------------------------------------------------------------------------
                elif options_sup == 'Regression':
                    try:
                        check_is_fitted(selected_model)
                    except NotFittedError as e:
                        st.error('Model has not been fitted. Model metrics cannot be calculated.')

                    # Begin Regression Loss Function Code ----------------------------------------------------------------------
                    with st.expander('Loss Functions'):
                        y_pred = selected_model.predict(X_test)
                        MSE = mean_squared_error(y_test, y_pred)
                        RMSE = root_mean_squared_error(y_test, y_pred)
                        MAE = mean_absolute_error(y_test, y_pred)
                        MAPE = mean_absolute_percentage_error(y_test, y_pred)

                        reg_metrics = pd.DataFrame([{
                            'Mean Squared Error': MSE,
                            'Root Mean Squared Error': RMSE,
                            'Mean Absolute Error': MAE,
                            'Mean Absolute Percentage Error': MAPE
                        }])


                        reg_metrics = reg_metrics.transpose().reset_index()
                        reg_metrics.columns = ['Loss Function', 'Value']

                        st.dataframe(reg_metrics, hide_index=True)
                    # End Regression Loss Function Code ------------------------------------------------------------------------

                    with st.expander('Expected vs Actual'):
                        def plot_predicted_vs_actual(y, y_pred, title="Predicted vs Actual"):
                            df = pd.DataFrame({
                                'Actual': y,
                                'Predicted': y_pred
                            })

                            fig = px.scatter(
                                df, x='Actual', y='Predicted',
                                title=title,
                                labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'},
                                opacity=0.7
                            )

                            # Add a perfect prediction line (y = x)
                            fig.add_shape(
                                type='line',
                                x0=min(y), y0=min(y),
                                x1=max(y), y1=max(y),
                                line=dict(color='red', dash='dash'),
                                name='Perfect Prediction'
                            )

                            fig.update_layout(showlegend=False)
                            return fig


                        # Example usage in Streamlit
                        st.title("Regression Model Evaluation")

                        st.plotly_chart(plot_predicted_vs_actual(y_test, y_pred))

                #End Regression Metrics Code -----------------------------------------------------------------------------------

                #Begin Feature Importance Code -----------------------------------------------------------------------
                with st.expander('Feature Importance (SHAP)'):
                    shap.initjs()
                    if 'shap_values' not in st.session_state:
                        st.session_state['shap_values'] = None
                        st.session_state['class_count'] = 0

                    if st.button('Generate SHAP Plots'):
                        try:
                            check_is_fitted(selected_model)

                            if hasattr(selected_model, "predict_proba"):
                                predict_fn = lambda x: selected_model.predict_proba(x)
                            else:
                                predict_fn = lambda x: selected_model.predict(x)

                            with st.spinner("Generating SHAP values..."):
                                model_type = type(selected_model)
                                test_sample = X_test[:50] if len(X_test) > 50 else X_test

                                if isinstance(X_train, np.ndarray):
                                    X_train = pd.DataFrame(X_train,
                                                           columns=[f'PC{i + 1}' for i in range(X_train.shape[1])],
                                                           index=st.session_state['X_train_index'])
                                    X_test = pd.DataFrame(X_test,
                                                          columns=[f'PC{i + 1}' for i in range(X_test.shape[1])],
                                                          index=st.session_state['X_test_index'])

                                background_data = X_train.sample(n=min(100, len(X_train)), random_state=42)

                                if model_type in [RandomForestClassifier, RandomForestRegressor,
                                                  HistGradientBoostingRegressor]:
                                    explainer = shap.TreeExplainer(selected_model)
                                elif model_type in [Ridge, LogisticRegression]:
                                    explainer = shap.LinearExplainer(selected_model, X_train)
                                elif model_type in [svm.SVC, svm.SVR, KNeighborsClassifier]:
                                    explainer = shap.KernelExplainer(predict_fn, background_data)
                                else:
                                    explainer = shap.Explainer(predict_fn, X_train)

                                shap_vals = explainer(test_sample)

                                # Save results to session state
                                st.session_state['shap_values'] = shap_vals

                                if isinstance(shap_vals, list):
                                    st.session_state['class_count'] = len(shap_vals)
                                elif hasattr(shap_vals, "values") and shap_vals.values.ndim == 3:
                                    st.session_state['class_count'] = shap_vals.values.shape[2]
                                else:
                                    st.session_state['class_count'] = 0

                        except NotFittedError:
                            st.error("Model is not fitted. SHAP explanations cannot be generated.")
                        except Exception as e:
                            st.error(f"SHAP explanation failed: {e}")

                    # Now outside the button, show the class selector if shap_values exist and multiple classes
                    if st.session_state['shap_values'] is not None:
                        shap_values = st.session_state['shap_values']

                        if st.session_state['class_count'] > 1:
                            class_idx = st.selectbox("Select class index for SHAP plots:",
                                                     options=list(range(st.session_state['class_count'])),
                                                     key='shap_class_idx')
                        else:
                            class_idx = 0

                        # Select correct shap_values to plot
                        if isinstance(shap_values, list):
                            shap_values_to_plot = shap_values[class_idx]
                        elif hasattr(shap_values, "values") and shap_values.values.ndim == 3:
                            shap_values_to_plot = shap.Explanation(
                                values=shap_values.values[:, :, class_idx],
                                base_values=shap_values.base_values[:, class_idx],
                                data=shap_values.data,
                                feature_names=shap_values.feature_names
                            )
                        else:
                            shap_values_to_plot = shap_values

                        st.subheader("SHAP Summary Plot")
                        fig_summary, ax_summary = plt.subplots()
                        shap.plots.beeswarm(shap_values_to_plot, show=False)
                        st.pyplot(fig_summary)

                        st.subheader("SHAP Bar Plot (Feature Importance)")
                        fig_bar, ax_bar = plt.subplots()
                        shap.plots.bar(shap_values_to_plot, show=False)
                        st.pyplot(fig_bar)
                #End Feature Importance Code -------------------------------------------------------------------------

            # End Model Metrics Code -------------------------------------------------------------------------------------------

    with pred_tab:
        pred_data = None
        pred_data = st.file_uploader('Upload a prediction data file', type='csv')

        if pred_data is not None:

            X_pred = pd.read_csv(pred_data)
            X_pred_original = deepcopy(X_pred)
            X_pred = X_pred.drop(columns=st.session_state['target'], errors='ignore')

            if data_form in ['Raw', 'Scaled']:
                X_pred = X_pred.select_dtypes(include='number')
                X_pred = X_pred.drop(columns=st.session_state['target'], errors='ignore')

                if data_form is 'Scaled':
                    scaler = st.session_state['sup_raw_scaler']
                    columns = X_train.drop(columns=st.session_state['target'], errors='ignore').columns
                    scaled_array = scaler.transform(X_pred[columns])
                    scaled_df = pd.DataFrame(scaled_array, columns=columns, index=X_pred.index)
                    X_pred = scaled_df


            else:
                encoder = st.session_state['sup_encoder']
                cat_features = st.session_state['cat_features']
                encoded_cat = encoder.transform(X_pred[cat_features])

                if hasattr(encoder, 'get_feature_names_out'):
                    encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_features),
                                              index=X_pred.index)


                numeric_df = X_pred.drop(columns=cat_features)

                # Step 4: Concatenate encoded + numeric features
                X_pred = pd.concat([numeric_df, encoded_df], axis=1)
                if data_form == 'Encoded & Scaled':
                    scaler = st.session_state['sup_encode_scaler']
                    columns = X_train.drop(columns=st.session_state['target'], errors='ignore').columns
                    encoded_scaled_array = scaler.transform(X_pred[columns])
                    encoded_scaled_df = pd.DataFrame(encoded_scaled_array, columns=columns, index=X_pred.index)
                    X_pred = encoded_scaled_df


            try:
                predictions = selected_model.predict(X_pred)

                #Adds predictions to original dataset
                pred_series = pd.Series(predictions, name=st.session_state['target'], index=X_pred_original.index)
                X_pred_with_preds = X_pred_original.copy()
                X_pred_with_preds[st.session_state['target']] = pred_series

                # Display and optionally allow download
                st.subheader("Predictions")

                view_pred = st.radio(label='Select Viewing Type',
                                     options=['Predictions Only', 'Predictions w/ Dataset'],
                                     captions=['Array of predictions', 'Predictions appended to original dataset'],
                                     horizontal=True)

                if view_pred is 'Predictions Only':
                    st.dataframe(predictions)
                else:
                    st.dataframe(X_pred_with_preds)

                # Optional: download button
                csv = X_pred_with_preds.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

            except ValueError as e:
                st.info(f"{e}")
                st.stop()




