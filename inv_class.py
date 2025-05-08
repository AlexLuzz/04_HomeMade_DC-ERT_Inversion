import numpy as np
from mesh_class import Mesh
from jacobian_class import Jacobian
from data_class import DataContainer
import pyvista as pv

class Inv:
    def __init__(self,  data_container):
        """
        Initialize the Inv object.

        Parameters:
        length (float): Length of the mesh along the x-axis.
        depth (float): Depth of the mesh along the z-axis.
        square_size (float): Size of each square in the mesh.
        data_container (DataContainer): The data container object containing electrode positions and protocol.
        """
        self.data_container = data_container
        self.mesh = None
        self.jacobian = None

    def set_mesh(self, length, depth, square_size):
        """
        Set the mesh parameters and create the mesh.

        Parameters:
        length (float): Length of the mesh along the x-axis.
        depth (float): Depth of the mesh along the z-axis.
        square_size (float): Size of each square in the mesh.
        resistivity (float): The background resistivity value.
        """
        self.mesh = Mesh(length, depth, square_size)

    def show_sensitivity(self):
        """
        Visualize the sensitivity matrix.
        """
        jacobian = Jacobian(self.mesh ,self.data_container)
        jacobian.show_sensitivity()

    def plot_electrodes(self):
        """
        Plot the electrode positions on the mesh.
        """
        if self.mesh is None:
            raise ValueError("Mesh is not set. Please set the mesh first.")
        
        electrode_positions = np.array(self.data_container.elecs_pos)

        # Add a zero y-coordinate to the 2D electrode positions to make them 3D
        if electrode_positions.shape[1] == 2:  # Check if positions are 2D
            electrode_positions = np.hstack((electrode_positions, np.zeros((electrode_positions.shape[0], 1))))

        # Convert electrode positions to PyVista PolyData
        elecs = pv.PolyData(electrode_positions)
        plotter = pv.Plotter()
        plotter.add_mesh(self.mesh.grid, show_edges=True, color='lightblue')
        plotter.add_points(elecs, color='red', point_size=10, render_points_as_spheres=True)
        plotter.show_axes()
        plotter.show_grid()
        plotter.view_xz()
        plotter.disable()
        plotter.show()