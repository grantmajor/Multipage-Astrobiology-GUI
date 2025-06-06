import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
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
# if 'data_file_data' in st.session_state:
#     data = st.session_state['data_file_data']
#
# if data is not None:
#     X_sup = data.select_dtypes(include=['int64', 'float64'])
#
#     st.divider()
#     st.subheader('Feature Scaling')
#
#     elements_sup = st.multiselect('Select the continuous explanatory variables to be scaled',
#                                          options = X_sup.columns,
#                                          default = X_sup.columns)
#
#     X_sup_scaled = X_sup[elements_sup]
#     X_sup = X_sup[elements_sup]
#
#     scaler = st.selectbox('Choose a feature scaler', options = ['Min-Max Scaling', 'Standardization'])
#     if scaler == 'Min-Max Scaling':
#         scaler = MinMaxScaler()
#     elif scaler == 'Standardization':
#         scaler = StandardScaler()
#
#     X_sup_scaled = scaler.fit_transform(X_sup_scaled)
#     X_sup_scaled = pd.DataFrame(X_sup_scaled, columns = elements_sup)
#
#     st.session_state['X_sup_scaled'] = X_sup_scaled
#
#     view_data = st.radio('**Preview Data**',
#                          ['Scaled Data', 'Unscaled Data', 'Compare'],
#                          captions = [
#                              'Show preview of scaled dataframe',
#                              'Show preview of unscaled data frame',
#                              'Preview both dataframes side-by-side'],
#                          horizontal = True
#                          )
#
#     if view_data == 'Scaled Data':
#         st.dataframe(X_sup_scaled.head())
#     elif view_data == 'Unscaled Data':
#         st.dataframe(X_sup.head())
#     else:
#         scale_col, unscale_col = columns(2, border = True)
#         with scale_col:
#             st.markdown("**Scaled Data**")
#             st.dataframe(X_sup_scaled.head())
#         with unscale_col:
#             st.markdown("**Unscaled Data**")
#             st.dataframe(X_sup.head())
#

#End Preprocessing Code
