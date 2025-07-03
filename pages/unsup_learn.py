import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import plotly.express as px


st.title('Unsupervised Learning')

#Checks to ensure that data is in the cache
if 'data_file_data' not in st.session_state:
    st.warning('Data not uploaded, models cannot be trained.')
else:
    data = st.session_state['data_file_data']


col1, col2 = st.columns(2)

#Construct Dimensionality Reduction and Clustering
if 'data_file_data' in st.session_state:
    with col1:
        st.subheader('**Here, the user can employ dimensionality reduction and clustering methods.**')
        st.divider()

        # Remove Columns that are Strings
        data_form = st.radio(label='Select Form of Data',
                             options=['Raw', 'Encoded', 'Scaled', 'Encoded & Scaled'],
                             horizontal=True,
                             captions=['Raw data',
                                       'Encoded data ',
                                       'Scaled data',
                                       'Encoded and scaled data'])

        y_train = st.session_state['y_train']
        y_test = st.session_state['y_test']

        if data_form == 'Raw':
            X_train = st.session_state['X_train']
            X_test = st.session_state['X_test']

            X_train = X_train.select_dtypes(include='number')
            X_test = X_test.select_dtypes(include='number')

            st.warning("Non-numerical features will be dropped when handling raw data")


        elif data_form == 'Encoded':
            if st.session_state['encoder_on']:
                X_train = st.session_state['X_train_encoded']
                X_test = st.session_state['X_test_encoded']
            else:
                st.warning("No encoder was selected in the preprocessing tab, proceededing with raw data")
                X_train = st.session_state['X_train']
                X_test = st.session_state['X_test']

                X_train = X_train.select_dtypes(include='number')
                X_test = X_test.select_dtypes(include='number')

        elif data_form == 'Scaled':
            X_train = st.session_state['X_train_scaled']
            X_test = st.session_state['X_test_scaled']

        else:
            X_train = st.session_state['X_train_encode_scaled']
            X_test = st.session_state['X_test_encode_scaled']


        options = st.selectbox(label='Select Dimensionality Reduction Method',
                               options=['Standard PCA',
                                        't-SNE'])

        # t-SNE Construction
        st.divider()
        # Set random state of the subsequent scripts
        np.random.seed(42)

        X = st.session_state['X_raw']
        y = st.session_state['y_raw']


        if options == 't-SNE':
            st.subheader('Define t-SNE Parameters')

            n_components = st.number_input('Insert Number of Components',
                                           min_value=2
                                           )
            perplexity = st.number_input('Insert Perplexity',
                                         min_value=2
                                         )
            tsne = TSNE(n_components,
                        random_state=42,
                        perplexity=perplexity,
                        n_jobs=-1,
                        method='exact',
                        max_iter=5000
                        )
            tsne_result = tsne.fit_transform(X)
            tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0],
                                           'tsne_2': tsne_result[:, 1]}
                                          )

            # DBSCAN
            st.divider()
            clusters = st.selectbox(label='Select Cluster Method',
                                    options=['Kmeans',
                                             'DBSCAN',
                                             'Target'])

            if clusters == 'Kmeans':
                st.subheader('Define K-means Parameters')
                n_clusters = st.number_input('Enter Number of Clusters',
                                             min_value=2
                                             )

                X_Kmeans = KMeans(n_clusters=n_clusters).fit(tsne_result)
                labels = X_Kmeans.labels_

            elif clusters == 'DBSCAN':
                st.subheader('Define DBSCAN Parameters')
                eps = st.number_input('Enter Eps',
                                      min_value=0.5
                                      )
                min_samples = st.number_input('Enter Minimum Samples',
                                              min_value=1
                                              )

                X_DBSCAN = DBSCAN(eps=eps, min_samples=min_samples).fit(tsne_result)
                DBSCAN_labels = X_DBSCAN.labels_
                DBSCAN_labels = DBSCAN_labels.astype(str)
                labels = [outlier.replace('-1', 'Outlier') for outlier in DBSCAN_labels]

            else:
                labels = y

            with col2:
                # Plot t-SNE Results
                st.subheader('t-Distributed Stochastic Neighbor Embedding')
                fig, ax = plt.subplots()
                fig = px.scatter(tsne_result_df,
                                 x='tsne_1',
                                 y='tsne_2',
                                 color=labels,
                                 title=options
                                 )
                fig.update_traces(
                    marker=dict(size=8,
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

        else:
            st.subheader('Define Standard PCA Parameters')


            n_components = st.number_input('Insert Number of Components',
                                           min_value=2
                                           )

            pca = PCA(n_components=n_components)
            pipe = Pipeline([('scaler', StandardScaler()),
                             ('pca', pca)])
            Xt = pipe.fit_transform(X)

            with col1:
                # DBSCAN
                st.divider()
                clusters = st.selectbox(label='Select Cluster Method',
                                        options=['Kmeans',
                                                 'DBSCAN',
                                                 'Target'])
                if clusters == 'Kmeans':
                    st.subheader('Define K-means Parameters')
                    n_clusters = st.number_input('Enter Number of Clusters',
                                                 min_value=2
                                                 )

                    X_Kmeans = KMeans(n_clusters=n_clusters).fit(Xt)
                    labels = X_Kmeans.labels_

                elif clusters == 'DBSCAN':
                    st.subheader('Define DBSCAN Parameters')
                    eps = st.number_input('Enter Eps',
                                          min_value=0.5
                                          )
                    min_samples = st.number_input('Enter Minimum Samples',
                                                  min_value=1
                                                  )

                    X_DBSCAN = DBSCAN(eps=eps, min_samples=min_samples).fit(Xt)
                    DBSCAN_labels = X_DBSCAN.labels_
                    DBSCAN_labels = DBSCAN_labels.astype(str)
                    labels = [outlier.replace('-1', 'Outlier') for outlier in DBSCAN_labels]

                else:
                    labels = y

            with col2:
                # Plot PCA Results
                st.subheader('Principal Component Analysis')
                fig, ax = plt.subplots()
                PCA_df = pd.DataFrame({'PCA_1': Xt[:, 0],
                                       'PCA_2': Xt[:, 1],
                                       'labels': labels},
                                      )
                fig = px.scatter(PCA_df,
                                 x='PCA_1',
                                 y='PCA_2',
                                 color=labels,
                                 title=options)
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
