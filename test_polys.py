# %%



# %%
import matplotlib.pyplot as plt
import numpy as np

def generate_concave_polygon(num_vertices):
    # Set the center and radius
    center = (0, 0)
    radius = 5

    # Generate vertices around a circle
    angles = np.linspace(0, 2*np.pi, num_vertices, endpoint=False)
    vertices_x = center[0] + radius * np.cos(angles)
    vertices_y = center[1] + radius * np.sin(angles)

    # Introduce concavity by modifying some vertices
    for i in range(4, num_vertices, 5):
        vertices_x[i] += 1.5
        vertices_y[i] += 1.5

    return list(zip(vertices_x, vertices_y))

# Example usage
num_vertices = 20
concave_polygon_vertices = generate_concave_polygon(num_vertices)

# %%

concave_polygon_vertices

# %%


# Plotting the concave polygon
polygon_x, polygon_y = zip(*concave_polygon_vertices)
plt.plot(polygon_x + (polygon_x[0],), polygon_y + (polygon_y[0],), color='black')
plt.scatter(polygon_x, polygon_y, color='red', label='Vertices')
plt.title('Concave Polygon with 20 Vertices')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()

# %%

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import triangulate
import matplotlib.pyplot as plt


def tessellate_polygon(vertices):
    polygon = Polygon(vertices)
    if not polygon.is_simple or not polygon.is_valid:
        raise ValueError("Invalid or self-intersecting polygon")

    triangles = []
    for triangle in triangulate(polygon):
        triangles.append(list(triangle.exterior.coords[:-1]))

    return triangles

# Example usage
polygon_vertices = concave_polygon_vertices
triangles = tessellate_polygon(polygon_vertices)

# Plotting the original polygon and the resulting triangles
polygon_x, polygon_y = zip(*polygon_vertices)
for triangle in triangles:
    triangle_x, triangle_y = zip(*triangle)
    plt.fill(triangle_x, triangle_y, alpha=0.5)

plt.plot(polygon_x + (polygon_x[0],), polygon_y + (polygon_y[0],), color='black')
plt.title('Tessellated Polygon')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
