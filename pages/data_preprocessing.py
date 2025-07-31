import numbers

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
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
    categorical_elements = st.multiselect("Select Categorical Explanatory Variables",
                                          options=non_num_elements,
                                          placeholder='Choose Option',
                                          default=st.session_state['cat_features']
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


    st.session_state.update({
        'X_raw' : X,
        'y_raw' : y,
    })

    # Begin Train Test Split Code
    train_proportion = st.number_input('Enter the Proportion of Data to be Allocated to Training.',
                                       min_value=0.0,
                                       value=0.75,
                                       step=0.01,
                                       format="%.2f", )
    st.session_state['train_size'] = train_proportion
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion)


    st.session_state.update({
        'X_train' : X_train,
        'X_test' : X_test,
        'y_train' : y_train,
        'y_test' : y_test
    })

    # End Train Test Split Code

    #Begin  Encoding Code
    st.divider()
    st.subheader('Encoding')

    encoding_selection = st.selectbox("Select Encoding Technique for Categorical Variables",
                 placeholder = "Choose Option",
                 options = ['None', 'One-Hot', 'Ordinal', 'Target (Supervised Only)'],
                 index = 0)

    st.session_state['encoder_on'] = False
    st.session_state['scaler_on'] = False

    # TODO: Encoder parameters

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

            #fits encoder to training data
            sup_encoder.fit(X_train[categorical_elements])
            unsup_encoder.fit(X[categorical_elements])

            #Encode training data
            X_train_encoded_cat = encode_data(sup_encoder, X_train, categorical_elements, columns = True)

            #Encoding testing data with the encoder fit on the training data
            X_test_encoded_cat = encode_data(sup_encoder, X_test, categorical_elements, columns = True)


            #Encoding data for unsupervised learning portion
            X_encoded_cat = encode_data(unsup_encoder, X, categorical_elements, columns = True)

            st.session_state['encoder_on'] = True

            #Ordinal Encoding
        elif encoding_selection == 'Ordinal':
            sup_encoder = OrdinalEncoder()
            unsup_encoder = copy.deepcopy(sup_encoder)


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

            st.session_state['encoder_on'] = True

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

            st.session_state['encoder_on'] = True

            X_train_encoded_cat.columns = X_train_encoded_cat.columns.astype(str)
            X_test_encoded_cat.columns = X_test_encoded_cat.columns.astype(str)

            X_encoded_cat.columns = X_encoded_cat.columns.astype(str)

        else:
            st.session_state['encoder_on'] = False

        # Checks that an encoder has been selected
        if st.session_state['encoder_on']:

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


    scaler = st.selectbox('Select a Feature Scaling Technique', options = ['None', 'Min-Max Scaling', 'Standardization'], index=0)
    if scaler == 'Min-Max Scaling':
        sup_scaler = MinMaxScaler()
        unsup_scaler = copy.deepcopy(sup_scaler)
        st.session_state['scaler_on'] = True
    elif scaler == 'Standardization':
        sup_scaler = StandardScaler()
        unsup_scaler= copy.deepcopy(sup_scaler)
        st.session_state['scaler_on'] = True


    def scale_data(scaler, data):
        """ Applies the specified, pre-fit scaler to the data

        :param scaler: A pre-fitted scaler to be applied to the data
        :param data: Array-like form of data, will be scaled
        :return: A dataframe containing a scaled version of the 'data' parameter
        """
        return pd.DataFrame(scaler.transform(data),
                            columns = data.columns,
                            index = data.index)

    if st.session_state['scaler_on']:
        if st.session_state['encoder_on']:

            sup_scaler.fit(X_train_encoded)
            unsup_scaler.fit(X_encoded)

            #scaling TTS split data
            X_train_encode_scaled = scale_data(sup_scaler, X_train_encoded)
            X_test_encode_scaled = scale_data(sup_scaler, X_test_encoded)

            #scaling non-TTS data
            X_encoded_scaled = scale_data(unsup_scaler, X_encoded)

            st.session_state.update({
            'X_test_encode_scaled': X_test_encode_scaled,
            'X_train_encode_scaled': X_train_encode_scaled,
            'X_encoded_scaled' : X_encoded_scaled
            })

        else:

            sup_scaler.fit(X_train[numerical_elements])
            unsup_scaler.fit(X[numerical_elements])

            X_train_scaled =  scale_data(sup_scaler, X_train[numerical_elements])
            X_test_scaled =  scale_data(sup_scaler, X_test[numerical_elements])

            X_scaled = scale_data(unsup_scaler, X[numerical_elements])

            st.session_state.update({
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled,
                'X_scaled' : X_scaled
            })

        view_data = st.radio('**Preview Data**',
                     ['Scaled Data', 'Unscaled Data', 'Compare'],
                     captions = [
                         'Show preview of scaled dataframe',
                         'Show preview of unscaled data frame',
                         'Preview both dataframes side-by-side'],
                     horizontal = True
                     )

    if st.session_state['encoder_on'] and st.session_state['scaler_on']:
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


    elif st.session_state['scaler_on']:
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

