import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

# Open one of your rasters (e.g., Elevation)
with rasterio.open('data\emb11Nclp.tif') as src:
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the raster background
    show(src, ax=ax, cmap='terrain', title='hmmm')
    
    # Overlay Presence (Green) and Absence (Red)
    # Assumes points_df has 'lon' and 'lat'
    # points_df[points_df.label == 1].plot(kind='scatter', x='lon', y='lat', 
    #                                     ax=ax, color='green', label='Presence', s=10)
    # points_df[points_df.label == 0].plot(kind='scatter', x='lon', y='lat', 
    #                                     ax=ax, color='red', label='Absence', s=10)
    plt.legend()
    plt.show()