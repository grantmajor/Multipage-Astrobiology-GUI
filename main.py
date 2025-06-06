import streamlit as st

#streamlit run main.py

st.set_page_config(
    page_title = 'AI² Tool',
    page_icon = 'assets/AI_2_Favicon.png',
    layout = 'wide')


about_page = st.Page('pages/about.py', title ='About', icon = ":material/info:")
data_page = st.Page('pages/data_preprocessing.py', title ='Data and Preprocessing', icon = ":material/bar_chart:")
unsup_page = st.Page('pages/unsup_learn.py', title ='Unsupervised Learning', icon = ":material/scatter_plot:")
sup_page = st.Page('pages/sup_learn.py', title ='Supervised Learning', icon = ":material/school:")
defn_page = st.Page('pages/defn_uses.py', title = 'Definitions and Use Cases', icon = ":material/dictionary:")


pg = st.navigation({'General': [about_page, data_page, defn_page],
                    'Models': [sup_page, unsup_page]})

pg.run()