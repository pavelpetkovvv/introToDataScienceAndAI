import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def visualise_k_means(K, clustering_data):
    kmeans = KMeans(n_clusters=K, random_state=0)
    cluster_assignments = kmeans.fit_predict(clustering_data)
    clustering_data['Cluster'] = cluster_assignments

    fig = px.scatter(clustering_data, x='phi', y='psi', color='Cluster',
                 title='K-Means Clustering Results', labels={'phi': 'Phi Angle', 'psi': 'Psi Angle'})

    fig.show()

def visualise_k_means_custom_initialization(K, clustering_data, custom_initialization):
    kmeans = KMeans(n_clusters=K, init=custom_initialization)
    cluster_assignments = kmeans.fit_predict(clustering_data)
    clustering_data['Cluster'] = cluster_assignments

    fig = px.scatter(clustering_data, x='phi', y='psi', color='Cluster',
                 title='K-Means Clustering Results', labels={'phi': 'Phi Angle', 'psi': 'Psi Angle'})

    fig.show()

def visualise_DBSCAN_and_outliers_barplot(eps, min_samples, data):
    clustering_data=data[['phi', 'psi']]
    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clustering_data)

    # Create and fit the DBSCAN model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_assignments = dbscan.fit_predict(X_scaled)

    # Add the cluster assignments to the DataFrame
    data['Cluster'] = cluster_assignments

    fig = px.scatter(data, x='phi', y='psi', color='Cluster',
                    title='DBSCAN Clustering Results', labels={'phi': 'Phi Angle', 'psi': 'Psi Angle'})

    fig.show()

    outliers = data[data['Cluster'] == -1]

    # Count the number of outliers for each residue name
    outlier_counts = outliers['residue name'].value_counts().reset_index()
    outlier_counts.columns = ['Residue Name', 'Number of Outliers']

    # Create a bar chart using Plotly Express
    fig = px.bar(outlier_counts, x='Residue Name', y='Number of Outliers',
                labels={'Residue Name': 'Residue Name', 'Number of Outliers': 'Number of Outliers'},
                title='Outliers by Residue Name')

    # Show the interactive bar chart
    fig.show()