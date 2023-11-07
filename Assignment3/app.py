import pandas as pd
import numpy as np
import plotly.express as px
import utils as ut
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Drawing a scatter plot to visualise all the data
data = pd.read_csv("data_assignment3.csv")
fig = px.scatter(data, x='phi', y='psi', color='chain', size_max=10, labels={'phi': 'Phi Angle', 'psi': 'Psi Angle'},
                 title='Phi vs. Psi Scatter Plot')
fig.show()

# Drawing a density heatmap to visualise all the data
fig = px.density_heatmap(data, x='phi', y='psi', nbinsx=180, nbinsy=180, color_continuous_scale='Inferno',
                         title='2D Histogram of Phi vs. Psi',
                         labels={'phi': 'Phi Angle', 'psi': 'Psi Angle'})
fig.show()

# Eyeballing the plots, there are 2 very distinctive clusters (top left part of the graphic),
# and some not so distinctive. In my oppinion the clusters are 4 and 7, 3 of them are on the
# left side of the graphic and the right may be considered either 1 or 4 separate clusters.
# It is a matter of oppinion. If considered one cluster It would not matter because the data
# is so spred that it does not corelate but if it is 4 clusters then each cluster has verry little
# data compared to the clusters on the left side of the graphic

clustering_data = data[['phi', 'psi']]

# Dividing the data into 2, 3 and 4 clusters and visualising it. In both cases the centroid initialization
# is random and the algorith determines the clusters relatively well

K = 2
ut.visualise_k_means(K, clustering_data)

K = 3
ut.visualise_k_means(K, clustering_data)

K = 4
ut.visualise_k_means(K, clustering_data)

# Dividing the data into 7 clusters and visualising it using centroid initialization. In this case the data is not
# divided correctly into clusters
K = 7
ut.visualise_k_means(K, clustering_data)

# Dividing the data into 7 clusters and visualising it but with some manual input. The centeroids are manually set 
# to be close to the center of the clusters that are easily distinguishable by the human eye
clustering_data = data[['phi', 'psi']]
custom_initialization = np.array([[-100, -140], [-60, -33], [-117, -170], [80,20], [144, 160], [60, -165], [170, -170]])
K = 7
ut.visualise_k_means_custom_initialization(K, clustering_data, custom_initialization)

# Create a torus in 3D space with the 2D Ramachandran points onto it
phi_angles = data['phi']
psi_angles = data['psi']

# Create a torus in 3D space
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, 2 * np.pi, 100)
u, v = np.meshgrid(u, v)
R, r = 100, 40  # Major and minor radii of the torus
x = (R + r * np.cos(v)) * np.cos(u)
y = (R + r * np.cos(v)) * np.sin(u)
z = r * np.sin(v)

# Create a 3D plot with the torus
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='gray', alpha=0.5)

# Map the 2D Ramachandran points onto the torus
scaled_phi = phi_angles / 180 * np.pi
scaled_psi = psi_angles / 180 * np.pi
x_mapped = (R + r * np.cos(scaled_psi)) * np.cos(scaled_phi)
y_mapped = (R + r * np.cos(scaled_psi)) * np.sin(scaled_phi)
z_mapped = r * np.sin(scaled_psi)

ax.scatter(x_mapped, y_mapped, z_mapped, c='blue', s=20)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Ramachandran Plot Mapped onto a Torus')

# Remove the comment from the line below for the torus with the mapped 
# Ramachandran to be displayed
# plt.show()

# DBSCAN
eps = 0.2
min_samples = 300
ut.visualise_DBSCAN_and_outliers_barplot(eps, min_samples, data)

eps = 0.2
min_samples = 200
ut.visualise_DBSCAN_and_outliers_barplot(eps, min_samples, data)

# I tried different values but could not get DBSCAN to discover more thant 3 clusters
# (it did but clearly the clusterisation was not good).
# 
# With eps = 0.2 and min_samples = 300 DBSCAN
# succeeds in discovering 2 clusters (the ones on the top left) that visibly have
# much more data in them than the other parts of the graphic and also are distingushable on
# the heatmap
# With eps = 0.2 and min_samples = 200 DBSCAN
# succeeds in discovering 3 clusters (the ones on the top left and one on the right).
# The right one is visible less denser than the left ones but it also can be distinguished
# on the heatmap although not as much as the other two.
# In both cases most of the outliers are of residue type (name) GLY


# Cluster the data that have residue type PRO
lys_data = data[data['residue name'] == 'PRO']

# Select the 'phi' and 'psi' columns for clustering
X = lys_data[['phi', 'psi']]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose DBSCAN parameters (adjust as needed)
eps = 0.6
min_samples = 100

# Create and fit the DBSCAN model
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
cluster_assignments = dbscan.fit_predict(X_scaled)

# Add the cluster assignments to the DataFrame for LYS residues
lys_data['Cluster'] = cluster_assignments

# Visualize the clusters or outliers for LYS residues
outliers = lys_data[lys_data['Cluster'] == -1]

fig = px.scatter(lys_data, x='phi', y='psi', color='Cluster',
                title='Clustering of LYS Residues', labels={'phi': 'Phi Angle', 'psi': 'Psi Angle'})

fig.show()