import numbers

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, OneHotEncoder, \
    OrdinalEncoder, LabelEncoder, TargetEncoder
from streamlit import columns
import copy

import hashlib

# Begin Data Upload Code
st.subheader('Data')
data_file = st.file_uploader('Upload a data file', type='csv', key='data_file')


def get_file_hash(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    return hashlib.sha256(file_bytes).hexdigest()


data = None

if data_file is not None:
    file_hash = get_file_hash(data_file)

    # Check if this is a new file based on hash (not just name)
    new_file_uploaded = (
            'data_file_hash' not in st.session_state or
            st.session_state['data_file_hash'] != file_hash
    )

    data = pd.read_csv(data_file)

    # Always update data
    st.session_state['data_file_data'] = data
    st.session_state['data_file_name'] = data_file.name
    st.session_state['data_file_hash'] = file_hash

    if new_file_uploaded:
        # Auto-reset feature selections for new dataset
        X_num = data.select_dtypes(include=np.number)
        X_cat = data.select_dtypes(exclude=np.number)

        st.session_state['num_features'] = list(X_num.columns)
        st.session_state['cat_features'] = []
        st.session_state['target'] = None

        st.info("New dataset detected. Feature selections have been reset.")

#runs when data is detected in the persistent cache

#End Data Upload Code

#Begin Preprocessing Code
if 'data_file_data' in st.session_state:
    data = st.session_state['data_file_data']

if data is not None:

    # Remove Columns that are Strings

    display_choice = st.radio('',
                              options=['Data',
                                       'Correlation Matrix'],
                              horizontal=True)

    if display_choice == 'Data':
        st.markdown('**Data Preview**')
        st.dataframe(data.head())
    if display_choice == 'Correlation Matrix':
        corr_matrix = data.select_dtypes(include=np.number).corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            title='Correlation Matrix',
            aspect='auto'
        )

        st.plotly_chart(fig, use_container_width=True)


    st.subheader('Data and Hyperparameter Selection')

    X = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    X_num = data.select_dtypes(include = np.number)
    X_cat = data.select_dtypes(exclude = np.number)

    if 'num_features' not in st.session_state:
        st.session_state['num_features'] = X_num.columns

    if 'cat_features' not in st.session_state:
        st.session_state['cat_features'] = []

    if 'target' not in st.session_state:
        st.session_state['target'] = None

    numerical_elements = st.multiselect("Select Numerical Explanatory Variables (default is all numerical columns)",
                                  options=X_num.columns,
                                  placeholder='Choose Option',
                                  default=st.session_state['num_features']
                                )

    if list(numerical_elements) != list(st.session_state['num_features']):
        st.session_state['num_features'] = numerical_elements


    non_num_elements = [col for col in data if col not in numerical_elements]
    valid_previous_cat = [
        col for col in st.session_state.get('cat_features', [])
        if col in non_num_elements
    ]
    categorical_elements = st.multiselect("Select Categorical Explanatory Variables",
                                          options=non_num_elements,
                                          placeholder='Choose Option',
                                          key='cat_selector'
                                          )

    if list(categorical_elements) != list(st.session_state['cat_features']):
        st.session_state['cat_features'] = categorical_elements


    elements = np.concatenate((numerical_elements, categorical_elements))
    y = data.dropna()

    # Filter out any already selected explanatory variables from the target options
    unselected_elements = [col for col in data if col not in elements]

    # Warn if all columns are selected as explanatory variables
    if len(unselected_elements) == 0:
        st.warning("All possible variables have been selected. Deselect a variable to choose a target.")
        st.stop()

    # Default to previous target if it exists in the available options
    if 'target' in st.session_state and st.session_state['target'] in unselected_elements:
        default_target = st.session_state['target']
    else:
        default_target = unselected_elements[0]  # fallback default

    # Let user select target
    target = st.selectbox(
        'Select Target',
        options=unselected_elements,
        index=unselected_elements.index(default_target)
    )

    # Save the selected target to session state
    st.session_state['target'] = target

    # Check for overlap with explanatory variables
    if target in elements:
        st.warning("Overlapping target and explanatory variables detected.")
        st.stop()

    # Assign X and y properly
    X = data[elements]
    y = data[target]

    # Check target type
    if isinstance(y.iloc[0], numbers.Number):
        st.session_state['target_is_number'] = True
    else:
        st.session_state['target_is_number'] = False

        if target is not None and target in elements:
            st.warning("Overlapping target and explanatory variables detected.")
            st.stop()

    st.subheader("Train Test Split")
    # Train-test split
    train_proportion = st.number_input(
        'Enter the Proportion of Data to be Allocated to Training.',
        min_value=0.0,
        value=0.75,
        step=0.01,
        format="%.2f",
    )

    #Begin Imputation Code
    st.divider()

    # Data Cleaning Section
    st.subheader('Data Cleaning')
    if X.isnull().values.any():
        st.warning("Missing values detected. Data cleaning is strongly recommended.")

        # Drop rows where target is missing — always safe
        elements = [col for col in elements if col in data.columns]
        if target not in data.columns:
            st.error(f"Target column '{target}' not found in data.")
        else:
            full_df = data[elements + [target]].dropna(subset=[target])

        # Select cleaning strategy
        imputer_choices = ['Drop Values', 'Simple Imputer']
        imputer_selector = st.selectbox(label='Select Imputation Method',
                                        options=imputer_choices)

        # Drop missing rows if selected
        if imputer_selector == 'Drop Values':
            full_df = full_df.dropna()

        # Extract X and y from cleaned full_df
        X = full_df[elements]
        y = full_df[target]

        st.session_state['train_size'] = train_proportion

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion)

        # Imputation if selected
        if imputer_selector == 'Simple Imputer':
            impute_strategy_map = {
                'Mean': 'mean',
                'Median': 'median',
                'Most Frequent': 'most_frequent'
            }

            impute_strategy = st.radio(label='Imputing Strategy',
                                       options=impute_strategy_map.keys(),
                                       horizontal=True)

            with st.spinner('Imputing Data'):
                imputer = SimpleImputer(missing_values=np.nan,
                                        strategy=impute_strategy_map[impute_strategy])
                unsup_imputer = copy.deepcopy(imputer)

                # Only fit on training data to avoid leakage
                X_train = imputer.fit_transform(X_train)
                X_test = imputer.transform(X_test)
                X = imputer.transform(X)

                # Check that imputation was successful
                if (not np.isnan(X_train).any()
                        and not np.isnan(X_test).any()
                        and not np.isnan(X).any()):
                    st.success('NaN values imputed successfully')
                else:
                    st.error('Imputing failed. NaN values still detected')

            # Restore column structure after imputation
            all_elements = np.concatenate((numerical_elements, categorical_elements))
            X_train = pd.DataFrame(X_train, columns=all_elements, index=y_train.index)
            X_test = pd.DataFrame(X_test, columns=all_elements, index=y_test.index)
            X = pd.DataFrame(X, columns=all_elements, index=y.index)
        else:
            # If drop was selected, keep DataFrame structure
            X_train = pd.DataFrame(X_train, columns=elements, index=y_train.index)
            X_test = pd.DataFrame(X_test, columns=elements, index=y_test.index)
            X = pd.DataFrame(X, columns=elements, index=y.index)

        # Update session state
        st.session_state.update({
            'X_train': X_train,
            'X_train_index' : X_train.index,
            'X_test': X_test,
            'X_test_index' : X_test.index,
            'y_train': y_train,
            'y_test': y_test,
            'X_raw': X,
            'y_raw': y

        })
    #Begin  Encoding Code
    st.subheader('Encoding')


    # Define the encoding options
    encoding_options = ['None', 'One-Hot', 'Ordinal', 'Target (Supervised Only)']


    # Display the selectbox using session_state as the key
    encoding_selection = st.selectbox(
        "Select Encoding Technique for Categorical Variables",
        options=encoding_options,
        key='encoding_choice'
    )


    #One Hot Encoding

    #checks if categorical elements have been selected in selectbox
    if categorical_elements:

        def encode_data(encoder, data, elements, columns=False):
            """ Takes data parameter and encodes the specified elements. Assumes encoder has been fit

            :param encoder: A pre-fit encoder to be used to transform data
            :param data: Array-like form of data to be transformed using the encoder
            :param elements: Array-like form of feature names, used to select the columns for encoding
            :param columns: Boolean check to determine if column names should be used
            :return: A dataframe with the newly encoded data.
            """
            if not columns:
                return pd.DataFrame(encoder.transform(data[elements]),
                                    index=data.index)
            else:
                return pd.DataFrame(encoder.transform(data[elements]),
                                    columns=encoder.get_feature_names_out(elements),
                                    index=data.index)

        if encoding_selection == 'One-Hot':
            sup_encoder = OneHotEncoder(sparse_output= False, handle_unknown='ignore')
            unsup_encoder = copy.deepcopy(sup_encoder)

            st.session_state['sup_encoder'] = sup_encoder

            #fits encoder to training data
            sup_encoder.fit(X_train[categorical_elements])
            unsup_encoder.fit(X[categorical_elements])

            #Encode training data
            X_train_encoded_cat = encode_data(sup_encoder, X_train, categorical_elements, columns = True)

            #Encoding testing data with the encoder fit on the training data
            X_test_encoded_cat = encode_data(sup_encoder, X_test, categorical_elements, columns = True)


            #Encoding data for unsupervised learning portion
            X_encoded_cat = encode_data(unsup_encoder, X, categorical_elements, columns = True)


            #Ordinal Encoding
        elif encoding_selection == 'Ordinal':
            sup_encoder = OrdinalEncoder()
            unsup_encoder = copy.deepcopy(sup_encoder)

            st.session_state['sup_encoder'] = sup_encoder

            sup_encoder.fit(X_train[categorical_elements])
            unsup_encoder.fit(X[categorical_elements])

            #Encode training data
            X_train_encoded_cat = encode_data(sup_encoder, X_train, categorical_elements, columns = True)

            #Encoding testing data with the encoder fit on the training data
            try:
                X_test_encoded_cat = encode_data(sup_encoder, X_test, categorical_elements, columns = True)
            except ValueError as e:
                st.error(f"Ordinal Encoding failed. There are likely categories in the testing set that were not seen in the "
                         f"training set. {e}")
                st.stop()

            X_encoded_cat = encode_data(unsup_encoder, X, categorical_elements, columns = True)

        #Target Encoding
        #TODO: Make it so that column names are retained after encoding
        elif encoding_selection == 'Target (Supervised Only)':
            sup_encoder = TargetEncoder()
            unsup_encoder = copy.deepcopy(sup_encoder)

            sup_encoder.fit(X_train[categorical_elements], y_train)
            unsup_encoder.fit(X[categorical_elements], y)

            # Encode training data
            X_train_encoded_cat = encode_data(sup_encoder, X_train, categorical_elements, columns = False)
            # Encoding testing data with the encoder fit on the training data
            X_test_encoded_cat = encode_data(sup_encoder, X_test, categorical_elements, columns = False)

            #Encode non-TTS split data
            X_encoded_cat = encode_data(unsup_encoder, X, categorical_elements, columns = False)

            X_train_encoded_cat.columns = X_train_encoded_cat.columns.astype(str)
            X_test_encoded_cat.columns = X_test_encoded_cat.columns.astype(str)

            X_encoded_cat.columns = X_encoded_cat.columns.astype(str)

        else:
            encoding_selection = 'None'


        # Checks that an encoder has been selected
        if encoding_selection != 'None' and len(categorical_elements) > 0:

            #Combine the encoded categorical variables with the numerical elements
            X_train_encoded = pd.concat([X_train[numerical_elements], X_train_encoded_cat], axis=1)
            X_test_encoded = pd.concat([X_test[numerical_elements], X_test_encoded_cat], axis=1)

            X_encoded = pd.concat([X[numerical_elements],X_encoded_cat], axis=1)


            if st.checkbox('Preview Encoded Data'):
                st.dataframe(X_train_encoded.head())



            st.session_state.update({
                'X_train_encoded' : X_train_encoded,
                'X_test_encoded' : X_test_encoded,
                'X_encoded' : X_encoded
            })


    else:
        st.warning("Encoding can only be used when categorical variables are selected.")
    #End Encoding Code

    #Feature Scaling
    st.subheader('Feature Scaling')

    scaler_options = ['None', 'Min-Max Scaling', 'Standardization']


    # Create the selectbox using the session_state value
    scaler = st.selectbox(
        'Select a Feature Scaling Technique',
        options=scaler_options,
        key='scaler_selection'
    )

    # Apply the selected scaler
    if scaler == 'Min-Max Scaling':
        sup_encode_scaler = MinMaxScaler()
        sup_raw_scaler = copy.deepcopy(sup_encode_scaler)

        unsup_encode_scaler = copy.deepcopy(sup_encode_scaler)
        unsup_scaler = copy.deepcopy(sup_encode_scaler)

    elif scaler == 'Standardization':
        sup_encode_scaler = StandardScaler()
        sup_raw_scaler = copy.deepcopy(sup_encode_scaler)

        unsup_encode_scaler = copy.deepcopy(sup_encode_scaler)
        unsup_scaler = copy.deepcopy(sup_encode_scaler)

    else:
        scaler = 'None'


    def scale_data(scaler, data):
        """ Applies the specified, pre-fit scaler to the data

        :param scaler: A pre-fitted scaler to be applied to the data
        :param data: Array-like form of data, will be scaled
        :return: A dataframe containing a scaled version of the 'data' parameter
        """

        return pd.DataFrame(scaler.transform(data),
                            columns = data.columns,
                            index = data.index)

    #TODO: Rename scaler to scaler_selection to maintain variable naming consistency
    if scaler != 'None':

        #Scaled Data
        sup_raw_scaler.fit(X_train[numerical_elements])
        unsup_scaler.fit(X[numerical_elements])

        #Transform scaled data
        X_train_scaled = scale_data(sup_raw_scaler, X_train[numerical_elements])
        X_test_scaled = scale_data(sup_raw_scaler, X_test[numerical_elements])
        X_scaled = scale_data(unsup_scaler, X[numerical_elements])

        #Upload scaled data to session state
        st.session_state.update({'X_train_scaled': X_train_scaled,
                                 'X_test_scaled': X_test_scaled,
                                 'X_scaled': X_scaled,
                                 'sup_raw_scaler' : sup_raw_scaler})


        #Scaled and Encoded Data
        if encoding_selection != 'None' and 'X_train_encoded' in st.session_state:

            #Fit scalers to encoded data

            X_train_encoded = st.session_state['X_train_encoded']
            X_test_encoded = st.session_state['X_test_encoded']
            X_encoded = st.session_state['X_encoded']
            sup_encode_scaler.fit(X_train_encoded)
            unsup_encode_scaler.fit(X_encoded)


            #Scaling encoded data
            X_train_encode_scaled = scale_data(sup_encode_scaler, X_train_encoded)
            X_test_encode_scaled = scale_data(sup_encode_scaler, X_test_encoded)
            X_encoded_scaled = scale_data(unsup_encode_scaler, X_encoded)

            #Uploading encoded and scaled data to session state
            st.session_state.update({
            'X_test_encode_scaled': X_test_encode_scaled,
            'X_train_encode_scaled': X_train_encode_scaled,
            'X_encoded_scaled' : X_encoded_scaled,
            'sup_encode_scaler': sup_encode_scaler
            })


        view_data = st.radio('**Preview Data**',
                     ['Scaled Data', 'Unscaled Data', 'Compare'],
                     captions = [
                         'Show preview of scaled dataframe',
                         'Show preview of unscaled data frame',
                         'Preview both dataframes side-by-side'],
                     horizontal = True
                     )


        if encoding_selection != 'None' and 'X_train_encoded' in st.session_state:
            if view_data == 'Scaled Data':
                st.dataframe(X_train_encode_scaled.head())
            elif view_data == 'Unscaled Data':
                st.dataframe(X_train_encoded.head())
            else:
                scale_col, unscale_col = columns(2, border = True)
                with scale_col:
                    st.markdown("**Scaled Data**")
                    st.dataframe(X_train_encode_scaled.head())
                with unscale_col:
                    st.markdown("**Unscaled Data**")
                    st.dataframe(X_train_encoded.head())


        else:
            if view_data == 'Scaled Data':
                st.dataframe(X_train_scaled.head())
            elif view_data == 'Unscaled Data':
                st.dataframe(X_train[numerical_elements].head())
            else:
                scale_col, unscale_col = columns(2, border = True)
                with scale_col:
                    st.markdown("**Scaled Data**")
                    st.dataframe(X_train_scaled.head())
                with unscale_col:
                    st.markdown("**Unscaled Data**")
                    st.dataframe(X_train[numerical_elements].head())
#End Scaling Code

#End Preprocessing Code

