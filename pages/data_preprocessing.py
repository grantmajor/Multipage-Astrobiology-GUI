import streamlit as st
import pandas as pd
import numpy as np



#Begin Data Upload Code
st.subheader('Data Upload')
data_file = None

if data_file is None and 'data' not in st.session_state:
    data_file = st.file_uploader('Upload a data file', type='csv', key = 'data_file')

#saves data and name of file to a persistent cache
if data_file is not None:
    st.session_state['data_file_data'] = pd.read_csv(data_file)
    st.session_state['data_file_name'] = data_file.name

#runs when data is detected in the persistent cache
if 'data_file_name' in st.session_state:
    st.success(f"Current dataset: {st.session_state.data_file_name}.")
    show_data = st.checkbox(label = 'Preview data')
    if show_data:
        st.dataframe(st.session_state['data_file_data'].head())
#End Data Upload Code


#Begin Preprocessing Code
st.divider()
st.subheader('Preprocessing')
st.text('Work in Progress')
#End Preprocessing Code
