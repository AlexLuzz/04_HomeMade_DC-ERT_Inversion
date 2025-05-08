import pyvista as pv
import numpy as np

class Mesh:
    def __init__(self, length, depth, square_size):
        """
        Initialize the Mesh object.

        Parameters:
        length (float): Length of the mesh along the x-axis.
        depth (float): Depth of the mesh along the z-axis.
        square_size (float): Size of each square in the mesh.
        """
        self.length = length
        self.depth = depth
        self.square_size = square_size
        self.grid = self.create_rectangular_mesh()
        self.resistivity = np.zeros(self.grid.number_of_points)

    def create_rectangular_mesh(self):
        """
        Create a 2D rectangular mesh along the x and z axes using PyVista.

        Returns:
        pv.StructuredGrid: A 2D rectangular mesh.
        """
        # Calculate the number of points along each axis
        x_points = int(self.length / self.square_size) + 1
        z_points = int(self.depth / self.square_size) + 1

        # Create a grid of points
        x = np.linspace(-self.length/2, self.length/2, x_points)
        z = np.linspace(0, -self.depth, z_points)
        xv, zv = np.meshgrid(x, z)
        y = np.zeros_like(xv)

        # Create the structured grid
        grid = pv.StructuredGrid(xv, y, zv)
        return grid

    def set_background_resistivity(self, value):
        """
        Set the background resistivity value for the entire mesh.

        Parameters:
        value (float): The background resistivity value.
        """
        self.resistivity.fill(value)
        self.grid['resistivity'] = self.resistivity

    def set_resistivity(self, background, zones):
        """
        Assign resistivity values to the mesh based on defined zones.

        Parameters:
        zones (list of dict): A list of zones with resistivity values and boundaries.
        """
        # Set the background resistivity value
        self.set_background_resistivity(background)

        # Assign resistivity values based on zones
        for zone in zones:
            xmin, xmax = zone['x_range']
            zmin, zmax = zone['z_range']
            value = zone['resistivity']
            
            # Find points within the zone
            points = self.grid.points
            mask = (points[:, 0] >= xmin) & (points[:, 0] <= xmax) & (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
            self.resistivity[mask] = value

        # Update resistivity values in the mesh
        self.grid['resistivity'] = self.resistivity

    def showMesh(self, scalars=None, cmap=None):
        """
        Visualize the mesh with resistivity values.
        """
        plotter = pv.Plotter()
        plotter.add_mesh(self.grid, scalars=scalars, show_edges=True, cmap=None, color='lightblue')
        plotter.show_axes()
        plotter.show_grid()
        plotter.view_xz()
        plotter.disable()
        plotter.show()
    
    def showRes(self):
        """
        Visualize the mesh with resistivity values.
        """
        self.showMesh(scalars='resistivity', cmap='viridis')

