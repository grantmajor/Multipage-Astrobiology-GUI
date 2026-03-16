import streamlit as st


st.title('About')
st.subheader("Teachable Artificial Intelligence for Astrobiology Investigations (AI$^{2}$)")
st.markdown(
    "This application is designed to allow astrobiologists to experiment with machine learning approaches including unsupervised and "
    "supervised methods using their own data."
    " This application will be especially useful for those who may be interested in applying machine learning but either don't know "
    "where to start or do not have the time to learn programming languages.")

st.markdown('Developed by Floyd Nichols and Grant Major')
st.markdown('email: floydnichols@vt.edu\n\n'
            'email: grantmajor@vt.edu')

st.divider()
st.subheader("Machine Learning Workflow")
with st.container(horizontal_alignment="center"):
    st.image("assets\ML_Steps.png")

with st.expander("Exploratory Data Analysis"):
    st.write("EDA...")
with st.expander("Data Preprocessing"):
    st.write("Data preprocessing...")
with st.expander("Model Selection"):
    st.write("Model selection...")
with st.expander("Hyperparameter Tuning"):
    st.write("Hyperparameter tuning...")
with st.expander("Model Evaluation"):
    st.write("Model evaluation...")