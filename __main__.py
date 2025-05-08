from inv_class import Inv
from data_class import DataContainer

# Create the DataContainer object
data_container = DataContainer()

# Add multiple electrode positions
electrode_positions = [(-6, -0.1), (-4, -0.1), (4, -0.1), (6, -0.1)]
data_container.add_elecs_pos(electrode_positions)

# Add protocol entries (quadrupoles)
protocol = [[0, 1, 2, 3]]
data_container.add_protocol(protocol)

# Define the length, depth, and square size
length = 20.0
depth = 7.0
square_size = 0.5

# Create the Inv object with the DataContainer
inv = Inv(data_container)

inv.set_mesh(length, depth, square_size)

# Define zones with resistivity values
zones = [{'x_range': (-3, 3), 'z_range': (-2, -1), 'resistivity': 500}]

# Assign resistivity values to the mesh
inv.mesh.set_resistivity(background=100, zones=zones)

# Visualize the mesh with resistivity values
#inv.mesh.showRes()

#inv.plot_electrodes()

# Visualize the sensitivity matrix
inv.show_sensitivity()