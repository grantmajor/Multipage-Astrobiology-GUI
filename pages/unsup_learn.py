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
        X = data.select_dtypes(include=['int64', 'float64'])

        st.subheader('Data and Hyperparameter Selection')

        X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        X = X.dropna()

        elements = st.multiselect("Select Explanatory Variables (default is all numerical columns):",
                                  X.columns,
                                  default=X.columns
                                  )

        y = data
        y = y.dropna()

        target = st.selectbox('Choose Target',
                              options=y.columns,
                              )

        options = st.selectbox(label='Select Dimensionality Reduction Method',
                               options=['Standard PCA',
                                        't-SNE'])

        # t-SNE Construction
        st.divider()
        # Set random state of the subsequent scripts
        np.random.seed(42)

        if options == 't-SNE':
            st.subheader('Define t-SNE Parameters')

            X = X[elements]  # Make prediction based on selected elements
            y = y[target]

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
            X = X[elements]  # Make prediction based on selected elements
            y = y[target]

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
