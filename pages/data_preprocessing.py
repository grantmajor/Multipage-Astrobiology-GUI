import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, OneHotEncoder, \
    OrdinalEncoder, LabelEncoder, TargetEncoder
from streamlit import columns

#Begin Data Upload Code
st.subheader('Data')
data_file = None
data = None

if data_file is None and 'data' not in st.session_state:
    data_file = st.file_uploader('Upload a data file', type='csv', key = 'data_file')

#saves data and name of file to a persistent cache
if data_file is not None:
    data = pd.read_csv(data_file)
    st.session_state['data_file_data'] = data
    st.session_state['data_file_name'] = data_file.name

#runs when data is detected in the persistent cache
if 'data_file_name' in st.session_state:
    st.success(f"Current dataset: {st.session_state.data_file_name}.")
    show_data = st.checkbox(label = 'Preview data')
    if show_data:
        st.dataframe(st.session_state['data_file_data'].head())

#End Data Upload Code


#Begin Preprocessing Code
if 'data_file_data' in st.session_state:
    data = st.session_state['data_file_data']

if data is not None:

    # Remove Columns that are Strings
    st.subheader('Data and Hyperparameter Selection')

    X = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    X_num = data.select_dtypes(include = np.number)
    X_cat = data.select_dtypes(exclude = np.number)


    numerical_elements = st.multiselect("Select Numerical Explanatory Variables (default is all numerical columns)",
                                  X_num.columns,
                                  placeholder='Choose Option',
                                  default=X_num.columns,
                                )

    #TODO: Make it so that categorical variables are not deselected when changes are detected in numerical_elements
    non_num_elements = [col for col in data if col not in numerical_elements]
    categorical_elements = st.multiselect("Select Categorical Explanatory Variables",
                                          non_num_elements,
                                          placeholder='Choose Option',
                                          )
    elements = np.concatenate((numerical_elements, categorical_elements))

    y = data.dropna()

    unselected_elements = [col for col in data if col not in elements]

    if len(unselected_elements) == 0:
        st.warning("All possible variables have been selected. Deselect a variable to choose a target.")

    target = st.selectbox('Choose Target',
                              options=unselected_elements,
                              placeholder='Choose Option'
                              )
    if target is not None:
        X = X[elements]
        y = y[target]

    if target in X:
        st.warning("Overlapping target and explanatory variables detected.")
        st.stop()

    # Begin Train Test Split Code
    train_proportion = st.number_input('Enter the Proportion of Data to be Allocated to Training.',
                                       min_value=0.0,
                                       value=0.75,
                                       step=0.01,
                                       format="%.2f", )
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
    # TODO: This code can be optimized

    #One Hot Encoding

    #checks if categorical elements have been selected in selectbox
    if categorical_elements:

        if encoding_selection == 'One-Hot':
            encoder = OneHotEncoder(sparse_output= False, handle_unknown='ignore')

            #fits encoder to training data
            encoder.fit(X_train[categorical_elements])

            #Encode training data
            X_train_encoded_cat = pd.DataFrame(encoder.transform(X_train[categorical_elements]),
                                               columns=encoder.get_feature_names_out(categorical_elements),
                                               index=X_train.index
                                               )

            #Encoding testing data with the encoder fit on the training data
            X_test_encoded_cat = pd.DataFrame(encoder.transform(X_test[categorical_elements]),
                                              columns=encoder.get_feature_names_out(categorical_elements),
                                              index=X_test.index
                                              )
            st.session_state['encoder_on'] = True

            #Ordinal Encoding
        elif encoding_selection == 'Ordinal':
            encoder = OrdinalEncoder()

            encoder.fit(X_train[categorical_elements])

            #Encode training data
            X_train_encoded_cat = pd.DataFrame(encoder.transform(X_train[categorical_elements]),
                                               columns=categorical_elements,
                                               index=X_train.index
                                               )

            #Encoding testing data with the encoder fit on the training data
            try:
                X_test_encoded_cat = pd.DataFrame(encoder.transform(X_test[categorical_elements]),
                                                columns=categorical_elements,
                                                index=X_test.index
                                                )
            except ValueError as e:
                st.error(f"Ordinal Encoding failed. There are likely categories in the testing set that were not seen in the "
                         f"training set.")
                st.stop()

            encoder.fit(X_train[categorical_elements])
            st.session_state['encoder_on'] = True

        #Target Encoding
        #TODO: Make it so that column names are retained after encoding
        elif encoding_selection == 'Target (Supervised Only)':
            encoder = TargetEncoder()

            encoder.fit(X_train[categorical_elements], y_train)

            # Encode training data
            X_train_encoded_cat = pd.DataFrame(encoder.transform(X_train[categorical_elements]),
                                               index=X_train.index
                                               )

            # Encoding testing data with the encoder fit on the training data
            X_test_encoded_cat = pd.DataFrame(encoder.transform(X_test[categorical_elements]),
                                              index=X_test.index
                                              )
            st.session_state['encoder_on'] = True

            X_train_encoded_cat.columns = X_train_encoded_cat.columns.astype(str)
            X_test_encoded_cat.columns = X_test_encoded_cat.columns.astype(str)



        else:
            st.session_state['encoder_on'] = False


        # Checks that an encoder has been selected
        if st.session_state['encoder_on']:

            #Combine the encoded categorical variables with the numerical elements
            X_train_encoded = pd.concat([X_train[numerical_elements], X_train_encoded_cat], axis=1)
            X_test_encoded = pd.concat([X_test[numerical_elements], X_test_encoded_cat], axis=1)

            if st.checkbox('Preview Encoded Data'):
                st.dataframe(X_train_encoded.head())


            st.session_state.update({
                'X_train_encoded' : X_train_encoded,
                'X_test_encoded' : X_test_encoded
            })

        else:
            st.info("An encoder has not been selected.")

    else:
        st.warning("Encoding can only be used when categorical variables are selected.")
    #End Encoding Code

    #Feature Scaling
    st.subheader('Feature Scaling')


    scaler = st.selectbox('Choose a feature scaler', options = ['Min-Max Scaling', 'Standardization'])
    if scaler == 'Min-Max Scaling':
        scaler = MinMaxScaler()
    elif scaler == 'Standardization':
        scaler = StandardScaler()


    if st.session_state['encoder_on']:
        scaler.fit(X_train_encoded)

        X_train_encode_scaled = pd.DataFrame(scaler.transform(X_train_encoded),
                                      columns = X_train_encoded.columns,
                                      index = X_train_encoded.index)

        X_test_encode_scaled = pd.DataFrame(scaler.transform(X_test_encoded),
                                     columns = X_test_encoded.columns,
                                     index = X_test_encoded.index)

        st.session_state.update({
        'X_test_encode_scaled': X_test_encode_scaled,
        'X_train_encode_scaled': X_train_encode_scaled
        })

    else:
        scaler.fit(X_train[numerical_elements])

        X_train_scaled =  pd.DataFrame(scaler.transform(X_train[numerical_elements]),
                                     columns = X_train[numerical_elements].columns,
                                     index = X_train[numerical_elements].index)

        X_test_scaled =  pd.DataFrame(scaler.transform(X_test[numerical_elements]),
                                     columns = X_test[numerical_elements].columns,
                                     index = X_test[numerical_elements].index)

        st.session_state.update({
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled
        })

    view_data = st.radio('**Preview Data**',
                 ['Scaled Data', 'Unscaled Data', 'Compare'],
                 captions = [
                     'Show preview of scaled dataframe',
                     'Show preview of unscaled data frame',
                     'Preview both dataframes side-by-side'],
                 horizontal = True
                 )

    if st.session_state['encoder_on']:
        if view_data == 'Scaled Data':
            st.dataframe(X_train_encode_scaled.head())
        elif view_data == 'Unscaled Data':
            st.dataframe(X_train_encode_scaled.head())
        else:
            scale_col, unscale_col = columns(2, border = True)
            with scale_col:
                st.markdown("**Scaled Data**")
                st.dataframe(X_train_encode_scaled.head())
            with unscale_col:
                st.markdown("**Unscaled Data**")
                st.dataframe(X_train[numerical_elements].head())
    else:
        if view_data == 'Scaled Data':
            st.dataframe(X_train_scaled.head())
        elif view_data == 'Unscaled Data':
            st.dataframe(X_train_scaled.head())
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

