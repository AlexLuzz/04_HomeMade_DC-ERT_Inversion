import numpy as np
import pyvista as pv
from scipy.special import roots_legendre
from scipy.special import ellipk, ellipe
from scipy.integrate import dblquad, quad

class Jacobian:
    def __init__(self, mesh, data):
        """
        Initialize the Jacobian object.

        Parameters:
        mesh (Mesh): The mesh object containing the grid and resistivity values.
        data (DataContainer): The data container object containing electrode positions and protocol.
        """
        self.mesh = mesh
        self.data = data

    def compute_potential(self, electrode_pos, current, epsilon=1e-12):
        """
        Compute the potential at each point in the grid due to a current injection at the electrode position.

        Parameters:
        electrode_pos (tuple): The (x, z) position of the electrode.
        current (float): The current injected at the electrode.
        epsilon (float): A small value to avoid division by zero.

        Returns:
        np.ndarray: The potential at each point in the grid.
        """
        x, z = self.mesh.grid.points[:, 0], self.mesh.grid.points[:, 2]
        ex, ez = electrode_pos
        distance = np.sqrt((x - ex)**2 + (z - ez)**2) + epsilon
        potential = current / (2 * np.pi * self.mesh.resistivity * distance)
        return potential

    def gauss_legendre_integration(self, func, a, b, n):
        """
        Perform Gauss-Legendre integration.

        Parameters:
        func (callable): The function to integrate.
        a (float): The lower limit of integration.
        b (float): The upper limit of integration.
        n (int): The number of sampling points.

        Returns:
        float: The result of the integration.
        """
        [x, w] = roots_legendre(n)
        t = 0.5 * (x + 1) * (b - a) + a
        return 0.5 * (b - a) * np.sum(w * func(t))
    
    def compute_jacobian_homogeneous_pole_pole(self):
        """
        Compute the Jacobian or sensitivity matrix for ERT measurements in a Two-dimensional homogeneous 
        half-space, using a bounded integration using elliptic integrals K and E. 
        Formula used from Friedel (2000), DOI: 10.1190/1.1444737, and explained in Thomas Gunther's 
        thesis, DOI: 10.3929/ethz-a-010385654.

        S(x, z) = (1 / 4π2){T1[(x - xA)(x - xM ) + z^2] + T2} (3.44, Gunter's thesis)
        with xA and xM the positions of the current and potential electrodes. 
        And T1 and T2 define as :
        T1 = 2/(ab^2*(a^2-b^2)^2)*[(a^2+b^2)*E(etha) - 2*b^2*K(etha)]
        T2 = 2 / (a *(a^2-b^2)^2)*[(a^2+b^2)*K(etha) - 2*a^2*E(etha)]

        a^2 and b^2 are maximum and minimum of the values (x - xA)^2 + z^2 and (x - xM)^2 + z^2
        abd etha = 1-b^2/a^2

        Parameters:
        current (float): The current injected at the electrodes.
        n_points_per_direction (int): The number of sampling points per direction for Gauss-Legendre integration.

        Returns:
        np.ndarray: The Jacobian or sensitivity matrix.
        """
        n_points = self.mesh.grid.number_of_points
        protocol = self.data.get_protocol()
        jacobian = np.zeros((len(protocol), n_points))

        for i, quadrupole in enumerate(protocol):
            A, B, M, N = quadrupole
            current_electrodes = [self.data.elecs_pos[A], self.data.elecs_pos[B]]
            potential_electrodes = [self.data.elecs_pos[M], self.data.elecs_pos[N]]

            x_A, z_A = current_electrodes[0]
            x_M, z_M = potential_electrodes[0]

            # Define the sensitivity function for homogeneous half-space
            def sensitivity_func(x, z):
                a2 = max((x - x_A)**2 + z**2, (x - x_M)**2 + z**2)
                b2 = min((x - x_A)**2 + z**2, (x - x_M)**2 + z**2)
                eta = 1 - b2 / a2
                K_eta = ellipk(eta)
                E_eta = ellipe(eta)
                T1 = 2 / (a2 * b2 * (a2 - b2)**2) * ((a2 + b2) * E_eta - 2 * b2 * K_eta)
                T2 = 2 / (a2 * (a2 - b2)**2) * ((a2 + b2) * K_eta - 2 * a2 * E_eta)
                term1 = (x - x_A) * (x - x_M) + z**2
                return (T1 * term1 + T2) / (4 * np.pi**2)

            # Compute the sensitivity for each point in the grid
            for j in range(n_points):
                x, z = self.mesh.grid.points[j, 0], self.mesh.grid.points[j, 2]
                jacobian[i, j] = sensitivity_func(x, z)*1000

        return jacobian


    def compute_jacobian_homogeneous(self, min_points=8, max_points=1000):
        """
        Compute the Jacobian for a homogeneous half-space using numerical integration.
        """
        n_points = self.mesh.grid.number_of_points
        protocol = self.data.protocol
        k = self.data.k
        jacobian = np.zeros((len(protocol), n_points))
        epsilon = 1e-2  # Small value to handle singularities

        for i, quadrupole in enumerate(protocol):
            # Extract electrode positions
            A, B, M, N = quadrupole
            x_A, z_A = self.data.elecs_pos[A]
            x_B, z_B = self.data.elecs_pos[B]
            x_M, z_M = self.data.elecs_pos[M]
            x_N, z_N = self.data.elecs_pos[N]

            # Define the sensitivity function for homogeneous half-space
            def sensitivity_func(x, z):
                ra = np.sqrt((x - x_A)**2 + (z - z_A)**2 + epsilon)
                rb = np.sqrt((x - x_B)**2 + (z - z_B)**2 + epsilon)
                rm = np.sqrt((x - x_M)**2 + (z - z_M)**2 + epsilon)
                rn = np.sqrt((x - x_N)**2 + (z - z_N)**2 + epsilon)
                return (1 / (4 * np.pi**2)) * ((1 / ra**3 - 1 / rb**3) * (1 / rm**3 - 1 / rn**3)) * k[i]

            def gauss_legendre_2d_integral(sensitivity_func, x_bounds, z_bounds, n_points=23):
                """
                Perform 2D integration using Gauss-Legendre quadrature.
                
                Args:
                    sensitivity_func: Function to integrate, f(x, z).
                    x_bounds: Tuple (a, b) for x integration limits.
                    z_bounds: Tuple (c, d) for z integration limits.
                    n_points: Number of quadrature points (default: 23).
                
                Returns:
                    Integral value.
                """
                # Gauss-Legendre quadrature points and weights
                points, weights = np.polynomial.legendre.leggauss(n_points)
                
                # Transform points and weights for x and z
                x_a, x_b = x_bounds
                z_a, z_b = z_bounds
                x_points = 0.5 * (x_b - x_a) * points + 0.5 * (x_b + x_a)  # Scale to [x_a, x_b]
                z_points = 0.5 * (z_b - z_a) * points + 0.5 * (z_b + z_a)  # Scale to [z_a, z_b]
                x_weights = 0.5 * (x_b - x_a) * weights
                z_weights = 0.5 * (z_b - z_a) * weights
                
                # 2D integration using nested loops
                integral = 0.0
                for i, x in enumerate(x_points):
                    for j, z in enumerate(z_points):
                        integral += x_weights[i] * z_weights[j] * sensitivity_func(x, z)
                
                return integral
            
            # Vectorize sensitivity computation for the entire grid
            x_points = self.mesh.grid.points[:, 0]
            z_points = self.mesh.grid.points[:, 2]

            # Calculate sensitivity for all points in the grid
            for j in range(n_points):
                x, z = x_points[j], z_points[j]

                # Dynamic integration points based on distance
                distance = np.sqrt((x - x_A)**2 + (z - z_A)**2)
                #n_points_adaptive = int(min_points + (max_points - min_points) * np.exp(-distance))

                # Numerical integration using dblquad for adaptive precision
                delta = 0.01  # Adjust based on your mesh resolution or electrode spacing
                # Compute integral using Gauss-Legendre quadrature
                jacobian[i, j] = gauss_legendre_2d_integral(
                    sensitivity_func,
                    x_bounds=(x - delta, x + delta),
                    z_bounds=(z - delta, z + delta),
                    n_points=200
                ) * 1000  # Convert to mV/V

        return jacobian
        

    def compute_jacobian_inhomogeneous(self, current=1, n_points_per_direction=23):
        """
        Compute the Jacobian or sensitivity matrix for ERT measurements in an inhomogeneous medium.

        Parameters:
        current (float): The current injected at the electrodes.
        n_points_per_direction (int): The number of sampling points per direction for Gauss-Legendre integration.

        Returns:
        np.ndarray: The Jacobian or sensitivity matrix.
        """
        n_points = self.mesh.grid.number_of_points
        protocol = self.data.get_protocol()
        jacobian = np.zeros((len(protocol), n_points))

        for i, quadrupole in enumerate(protocol):
            A, B, M, N = quadrupole
            A_elecs = [self.data.elecs_pos[A], self.data.elecs_pos[B]]
            U_elecs = [self.data.elecs_pos[M], self.data.elecs_pos[N]]

            # Compute potentials for current electrodes
            potential_A = self.compute_potential(A_elecs[0], current)
            potential_B = self.compute_potential(A_elecs[1], -current)

            # Compute the potential difference at the potential electrodes
            potential_M = self.compute_potential(U_elecs[0], 1)
            potential_N = self.compute_potential(U_elecs[1], 1)

            # Define the sensitivity function for inhomogeneous medium
            def sensitivity_func(x):
                return (potential_M - potential_N) * (potential_A - potential_B)

            # Perform Gauss-Legendre integration for each point in the grid
            for j in range(n_points):
                jacobian[i, j] = self.gauss_legendre_integration(lambda t: sensitivity_func(t)[j], 0, 1, n_points_per_direction)

        return jacobian

    def show_sensitivity(self, homogeneous=True):
        """
        Visualize a measurement sensitivity matrix.

        Parameters:
        homogeneous (bool): If True, compute the Jacobian for a homogeneous half-space. If False, compute for an inhomogeneous medium.
        """
        if homogeneous:
            jacobian = self.compute_jacobian_homogeneous()
        else:
            jacobian = self.compute_jacobian_inhomogeneous()
        
        plotter = pv.Plotter()
        plotter.add_mesh(self.mesh.grid, scalars=jacobian.flatten(), show_edges=False, 
                         cmap='bwr', color='white', clim=[-0.01, 0.01])
        plotter.show_axes()
        plotter.show_grid()
        plotter.view_xz()
        plotter.disable()
        plotter.set_background("lightgrey")
        plotter.show()

def test_jacobian():
    """
    Test function for the Jacobian class.
    """
    # Create dummy mesh and data objects with necessary attributes
    class DummyMesh:
        def __init__(self):
            self.grid = pv.UniformGrid()
            self.grid.points = np.array([[0, 0, 0], [1, 0, 1], [2, 0, 2]])
            self.resistivity = 100
            self.grid.number_of_points = len(self.grid.points)

    class DummyDataContainer:
        def __init__(self):
            self.elecs_pos = [(0, 0), (1, 1), (2, 2), (3, 3)]
        
        def get_protocol(self):
            return [(0, 1, 2, 3)]

    mesh = DummyMesh()
    data = DummyDataContainer()
    jacobian = Jacobian(mesh, data)

    # Test compute_jacobian_homogeneous_pole_pole method
    result = jacobian.compute_jacobian_homogeneous()
    print("Jacobian (homogeneous pole-pole):", result)

if __name__ == "__main__":
    test_jacobian()