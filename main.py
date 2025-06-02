import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#streamlit run main.py

st.set_page_config(
    page_title = 'AI² Tool',
    page_icon = 'assets/AI_2_Favicon.png',
    layout = 'wide')

#TODO: Add icons for each page
about_page = st.Page('pages/about.py', title ='About')
data_page = st.Page('pages/data_preprocessing.py', title ='Data and Preprocessing')
unsup_page = st.Page('pages/unsup_learn.py', title ='Unsupervised Learning')
sup_page = st.Page('pages/sup_learn.py', title ='Supervised Learning')
defn_page = st.Page('pages/defn_uses.py', title = 'Definitions and Use Cases')


pg = st.navigation({'General': [about_page, data_page, defn_page],
                    'Models': [sup_page, unsup_page]})

pg.run()