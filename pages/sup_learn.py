import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay, \
    mean_absolute_percentage_error, mean_absolute_error, root_mean_squared_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils.validation import check_is_fitted
import seaborn as sns

st.title('Supervised Learning')



#Checks to ensure that data is in the cache
if 'data_file_data' not in st.session_state:
    st.warning('Data not uploaded, models cannot be trained.')
else:
    data = st.session_state['data_file_data']

model_tab, pred_tab = st.tabs(['Model', 'Predictions'])



# Begin Model Training Code --------------------------------------------------------------------------------------------

if 'data_file_data' in st.session_state:
    with model_tab:
        col1, col2 = st.columns(spec=2, gap='small', vertical_alignment='top')
        with col1:

            options_sup = st.selectbox(label='Select Prediction Type',
                                       options=['Classification',
                                                'Regression'])

            st.divider()

            data_form = st.radio(label = 'Select Form of Data',
                                 options = ['Raw', 'Encoded', 'Scaled', 'Encoded & Scaled'],
                                 horizontal = True,
                                 captions = ['Raw data',
                                             'Encoded data ',
                                             'Scaled data',
                                             'Encoded and scaled data'])

            y_train = st.session_state['y_train']
            y_test = st.session_state['y_test']

            if data_form == 'Raw':
                X_train = st.session_state['X_train']
                X_test = st.session_state['X_test']

                X_train = X_train.select_dtypes(include = 'number')
                X_test = X_test.select_dtypes(include = 'number')

                st.warning("Non-numerical features will be dropped when handling raw data")


            elif data_form == 'Encoded':
                if st.session_state['encoder_on']:
                    X_train = st.session_state['X_train_encoded']
                    X_test = st.session_state['X_test_encoded']
                else:
                    st.warning("No encoder was selected in the preprocessing tab, proceeding with raw data")
                    X_train = st.session_state['X_train']
                    X_test = st.session_state['X_test']

                    X_train = X_train.select_dtypes(include='number')
                    X_test = X_test.select_dtypes(include='number')

            elif data_form == 'Scaled':
                X_train = st.session_state['X_train_scaled']
                X_test = st.session_state['X_test_scaled']

            else:
                if st.session_state['encoder_on']:
                    X_train = st.session_state['X_train_encoded_scaled']
                    X_test = st.session_state['X_test_encoded_scaled']
                else:
                    st.warning("No encoder was selected in the preprocessing tab, proceeding with scaled data")
                    X_train = st.session_state['X_train_scaled']
                    X_test = st.session_state['X_test_scaled']



            #Begin Regression Code -----------------------------------------------------------------------------------------

                # Begin HistGradBoost --------------------------------------------------------------------------------------

            def get_hgbrt_model():

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
                alpha_value = st.number_input(label='Input Alpha Value',
                                              min_value=0.0,
                                              value=1.0,
                                              step=0.01,
                                              format='%.2f')

                return  Ridge(alpha=alpha_value)
            # End Ridge Code -------------------------------------------------------------------------------------------

            def get_svr_model():
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
                                    value = 3,
                                    step = 1,
                                    min_value =1)
                degree = 3

                epsilon_value = st.number_input('Input epsilon value',
                                                value=0.1,
                                                step=0.01,
                                                min_value=0.01)
                return svm.SVR(C=c_value,
                               kernel=kernel_selection,
                               degree=degree,
                               epsilon=epsilon_value)


            # End SVM Code -----------------------

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
                # Sets degree to a default value in case kernel_type isn't polynomial and thus degree isn't declared
                degree = 3

                return svm.SVC(C=c_value,
                               kernel=kernel_selection,
                               degree=degree)

            # End SVM Code ---------------------------------------------------------------------------------------------

            # Begin k-NN Code ------------------------------------------------------------------------------------------
            def get_knn_model():
               # Selecting k-Value
               k_value = st.number_input('Input K Value.',
                                        min_value=1,
                                        value=1)
               return KNeighborsClassifier(n_neighbors=k_value)
            # End k-NN Code --------------------------------------------------------------------------------------------

            # Begin Random Forest Classifier Code ----------------------------------------------------------------------
            def get_random_forest_class_model():
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
            def get_logistic_regression_model():

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

                try:
                    selected_model.fit(X_train, y_train)
                except ValueError as e:
                    st.error(f"ValueError caught, ensure target variable is composed of discrete classes. {e}")
                    st.stop()

            # End Model Training Code ------------------------------------------------------------------------------------------

            # Begin Model Metrics Code -----------------------------------------------------------------------------------------
            with col2:
                st.subheader("Model Performance Metrics")
                try:
                    y_pred = selected_model.predict(X_test)
                except NotFittedError:
                    selected_model.fit(X_train, y_train)


                # Begin Cross Validation Code --------------------------------------------------------------------------
                # with st.expander('Cross Validation'):
                #     if 'cv_confirmation' not in st.session_state:
                #         st.session_state['cv_confirmation'] = False
                #
                #     if 'cv_ran' not in st.session_state:
                #         st.session_state['cv_ran'] = False
                #
                #     if not st.session_state['cv_ran'] and not st.session_state['cv_confirmation']:
                #         st.warning("Cross validation can be computationally expensive. The number of folds has "
                #                     "been limited to 10.")
                #         if st.button("I understand"):
                #             st.session_state['cv_confirmation'] = True
                #             st.rerun()
                #
                #     else:
                #         cv_folds = st.number_input('Enter number of cross validation folds',
                #                                    min_value = 2,
                #                                    max_value = 10,
                #                                    step = 1)
                #
                #         if st.button('Begin Cross Validation'):
                #             cv_scores = cross_validate(st.session_state['unfitted_model'],
                #                                        X_train, y_train,
                #                                        cv = cv_folds)
                #         #TODO: Figure out what scores to output from cross validation
                # End Cross Validation Code ---------------------------------------------------------------------------

                if options_sup == 'Classification':
                    try:
                        check_is_fitted(selected_model)
                    except NotFittedError as e:
                        st.warning(f'Model has not been fitted to data. Metrics cannot be shown. {e}')


                    # Begin Classification Report Code--------------------------------------------------------------------------
                    class_report = classification_report(y_test, y_pred, output_dict = True)
                    with st.expander('Classification Report'):
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

                    y_pred = selected_model.predict(X_test)
                    MSE = mean_squared_error(y_test, y_pred)
                    RMSE = root_mean_squared_error(y_test, y_pred)
                    MAE = mean_absolute_error(y_test, y_pred)
                    MAPE = mean_absolute_percentage_error(y_test, y_pred)

                    with st.expander('Loss Functions'):
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

                #End Regression Metrics Code -----------------------------------------------------------------------------------

            # End Model Metrics Code -------------------------------------------------------------------------------------------

    with pred_tab:
        st.text('Work in progress')

        pred_data = st.file_uploader('Upload a prediction data file', type = 'csv')
