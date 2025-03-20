import numpy as np
import cartopy.crs as ccrs

# Define source (lon/lat) and target (Mercator) projections
source_crs = ccrs.PlateCarree()  # (lon, lat) input
target_crs = ccrs.Mercator()     # Convert to Mercator projection

# Define a grid of longitudes and latitudes
lon = np.array([-120, -110, -100, -90])
lat = np.array([30, 40, 50, 60])

# Create a meshgrid of coordinates
lon_grid, lat_grid = np.meshgrid(lon, lat)  # Shape: (4, 4)

# Transform the entire grid to the Mercator projection
transformed = target_crs.transform_points(source_crs, lon_grid, lat_grid)

# Extract the transformed x (longitude) and y (latitude) coordinates
x_transformed = transformed[..., 0]  # First column is x
y_transformed = transformed[..., 1]  # Second column is y

print("Original Longitude Grid:\n", lon_grid)
print("Original Latitude Grid:\n", lat_grid)
print("Transformed X Grid (Mercator):\n", x_transformed)
print("Transformed Y Grid (Mercator):\n", y_transformed)
