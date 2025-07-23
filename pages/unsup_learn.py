import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import plotly.express as px



st.title('Unsupervised Learning')

#Checks to ensure that data is in the cache
if 'data_file_data' not in st.session_state:
    st.warning('Data not uploaded, models cannot be trained.')
    st.stop()
else:
    data = st.session_state['data_file_data']


col1, col2 = st.columns(2)

#Construct Dimensionality Reduction and Clustering
if 'data_file_data' in st.session_state:
    random_state = 42
    with col1:

        data_choices = ['Raw', 'Scaled']

        if st.session_state['encoder_on']:
            data_choices.append('Encoded')
            data_choices.append('Encoded & Scaled')


        data_form = st.radio(label='Select Form of Data',
                             options=['Raw', 'Encoded', 'Scaled', 'Encoded & Scaled'],
                             horizontal=True,
                             captions=['Raw data',
                                       'Encoded data ',
                                       'Scaled data',
                                       'Encoded and scaled data'])

        if data_form == 'Raw':
            X = st.session_state['X_raw']
            X = X.select_dtypes(include='number')
            st.warning("Non-numerical features will be dropped when handling raw data")

        elif data_form == 'Encoded':
            X = st.session_state['X_encoded']

        elif data_form == 'Scaled':
            X = st.session_state['X_scaled']

        else:
            X = st.session_state['X_encoded_scaled']

        y = st.session_state['y_raw']
        # Begin Dimensionality Reduction Method Code


        # Begin t-SNE Code

        def get_tsne_model():
            """ creates t-SNE model with user defined parameters

                Takes user input for the following model hyperparameters: number of components and perplexity
                and returns a t-SNE model with the user-specified hyperparameters

               :return: A t-SNE model with user-defined parameters
            """
            n_components = st.number_input('Insert Number of Components',
                                           min_value=2
                                           )
            perplexity = st.number_input('Insert Perplexity',
                                         min_value=2
                                         )
            return TSNE(n_components,
                        random_state=random_state,
                        perplexity=perplexity,
                        n_jobs=-1,
                        method='exact',
                        max_iter=5000
                        )


        # End t-SNE Code

        # Begin Standard PCA Code
        def get_pca_model():
            """ creates Standard PCA model with user defined parameters

                Takes user input for the following model hyperparameters: number of components,
                and returns a PCA model with the user-specified hyperparameters

               :return: A PCA model with user-defined parameters
            """
            n_components = st.number_input('Insert Number of Components',
                                           min_value=2
                                           )

            return PCA(n_components=n_components,
                      random_state=random_state)

        # End Standard PCA Code

        # Begin Multidimensional Scaling Code
        def get_mds_model():
            """ creates MDS model with user defined parameters

                Takes user input for the following model hyperparameters: number of components,
                and returns an MDS model with the user-specified hyperparameters

               :return: An MDS model with user-defined parameters
            """
            n_components = st.number_input('Insert Number of Components',
                                           min_value=2
                                           )
            return MDS(n_components=n_components,
                      random_state=random_state)


        def split_fit_store(model, X):
            """ Fits model on the training set of X and stores output in Streamlit session state

                Passes data to train test split. Model is fitted on X_train and is then used to transform X_train and
                X_test. This newly transformed data is stored in Streamlit's session state to allow for later use.

               :param model: A dimensionality reduction model that allows for fitting on X_train
               :param X: A dataframe containing data that is to be transformed by the model
            """
            try:
                train_proportion = st.session_state['train_size']
            except KeyError as e:
                print('Training proportion was not set in data and preprocessing')
                train_proportion = 0.75

            X_train, X_test = train_test_split(X, train_size=train_proportion)
            X_train = model.fit_transform(X_train)
            X_test = model.transform(X_test)
            st.session_state.update({'X_train_red' : X_train,
                                     'X_test_red' : X_test})


        model_display_names = {'t-SNE' : get_tsne_model,
                               'Standard PCA' : get_pca_model,
                               'Multidimensional Scaling' : get_mds_model}

        st.subheader("Dimensionality Reduction")
        dim_reduction_method = st.selectbox(label='Select Dimensionality Reduction Method',
                                            options=['Standard PCA',
                                                     't-SNE',
                                                     'Multidimensional Scaling'])

        if dim_reduction_method == 't-SNE':
            tsne = get_tsne_model()
            X_transformed = tsne.fit_transform(X)
            tsne_df = pd.DataFrame({'tsne_1': X_transformed[:, 0],
                                   'tsne_2': X_transformed[:, 1]}
                                   )
        elif dim_reduction_method == 'Standard PCA':
            pca = get_pca_model()
            split_fit_store(pca, X)
            X_transformed = pca.fit_transform(X)

        elif dim_reduction_method == 'Multidimensional Scaling':
            mds = get_mds_model()
            X_transformed = mds.fit_transform(X)
            mds_df = pd.DataFrame({"MDS_1": X_transformed[:, 0],
                                   'MDS_2': X_transformed[:, 1]})



    # End Dimensionality Reduction Code

        st.divider()
        st.subheader("Clustering Techniques")
        clustering_method = st.selectbox(label='Select Cluster Method',
                                        options=['Kmeans',
                                                 'DBSCAN',
                                                 'Target',
                                                 'Gaussian Mixture'])

        if clustering_method == 'Kmeans':
            n_clusters = st.number_input('Enter Number of Clusters',
                                         min_value=2
                                         )

            X_Kmeans = KMeans(n_clusters=n_clusters,
                              random_state=random_state).fit(X_transformed)
            labels = X_Kmeans.labels_

        elif clustering_method == 'DBSCAN':
            eps = st.number_input('Enter Eps',
                                  min_value=0.5
                                  )
            min_samples = st.number_input('Enter Minimum Samples',
                                          min_value=1
                                          )

            X_DBSCAN = DBSCAN(eps=eps,
                              min_samples=min_samples).fit(X_transformed)
            DBSCAN_labels = X_DBSCAN.labels_
            DBSCAN_labels = DBSCAN_labels.astype(str)
            labels = [outlier.replace('-1', 'Outlier') for outlier in DBSCAN_labels]

        elif clustering_method == 'Target':
            labels = y

        else:
            n_components = st.number_input('Enter number of components',
                                           min_value=1
                                           )

            cov_type_map = {'Full' : 'full',
                            'Tied' : 'tied',
                            'Diagonal' : 'diag',
                            'Spherical' : 'spherical'}


            cov_type_choice = st.selectbox('Select type of covariance',
                                              options=cov_type_map.keys(),
                                              index=0)


            cov_type_selection = cov_type_map[cov_type_choice]

            X_GM = GaussianMixture(n_components=n_components,
                                   covariance_type=cov_type_selection,
                                   random_state=random_state).fit(X)

            labels = X_GM.predict(X).astype(str)




    with col2:
        # Plot t-SNE Results
        if dim_reduction_method == 't-SNE':
            st.subheader('t-Distributed Stochastic Neighbor Embedding')
            fig, ax = plt.subplots()
            fig = px.scatter(tsne_df,
                             x='tsne_1',
                             y='tsne_2',
                             color=labels,
                             title=dim_reduction_method
                             )
            fig.update_traces(
                marker=dict(size=12,
                            line=dict(width=2,
                                      color='Black')
                            )
            )
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_aspect('auto')
            ax.legend('Cluster',
                      bbox_to_anchor=(0.8, 0.95),
                      loc=2,
                      borderaxespad=0.0)
            st.plotly_chart(fig, use_container_width=True)

        # Plot PCA Results
        elif dim_reduction_method == 'Standard PCA':
            st.subheader('Principal Component Analysis')
            fig, ax = plt.subplots()
            PCA_df = pd.DataFrame({'PCA_1': X_transformed[:, 0],
                                   'PCA_2': X_transformed[:, 1],
                                   'labels': labels},
                                  )
            fig = px.scatter(PCA_df,
                             x='PCA_1',
                             y='PCA_2',
                             color=labels,
                             title=dim_reduction_method)
            fig.update_traces(
                marker=dict(size=12,
                            line=dict(width=2,
                                      color='Black')
                            )
            )
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_aspect('auto')
            ax.legend(bbox_to_anchor=(0.8, 0.95),
                      loc=2,
                      borderaxespad=0.0)
            st.plotly_chart(fig, use_container_width=True)

            # Define and Plot Explained Variance Ratio
            fig, ax = plt.subplots()
            exp_var_pca = pca.explained_variance_ratio_
            fig = px.bar(exp_var_pca,
                         x=range(0, len(exp_var_pca)),
                         y=exp_var_pca,
                         title='PCA Explained Variance Ratio')

            fig.update_traces(
                marker=dict(color='grey',
                            line=dict(width=3,
                                      color='Black')
                            )
            )

            fig.update_layout(
                xaxis_title='Principal Component Index',
                yaxis_title='Explained Variance Ratio'
            )
            ax.set_aspect('auto')
            ax.legend(bbox_to_anchor=(0.8, 0.95),
                      loc=2,
                      borderaxespad=0.0)
            st.plotly_chart(fig, use_container_width=True)

        elif dim_reduction_method == 'Multidimensional Scaling':
            st.subheader('Multidimensional Scaling')
            fig, ax = plt.subplots()
            fig = px.scatter(mds_df,
                             x='MDS_1',
                             y='MDS_2',
                             color = labels,
                             title=dim_reduction_method
                             )
            fig.update_traces(marker = dict(size=12,
                                            line=dict(width=2,
                                                      color='Black')))
            ax.set_xlabel('MD 1')
            ax.set_ylabel('MD 2')
            ax.set_aspect('auto')
            ax.legend('Cluster',
                      bbox_to_anchor=(0.8, 0.95),
                      loc=2,
                      borderaxespad=0.0)
            st.plotly_chart(fig, use_container_width=True)

