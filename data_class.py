import numpy as np

class DataContainer:
    def __init__(self, elecs_pos=None, protocol=None):
        """
        Initialize the DataContainer object.

        Parameters:
        elecs_pos (list of tuple): Positions of the electrodes.
        protocol (list of list): Measurement protocol as a list of quadrupoles (A, B, M, N).
        """
        self.elecs_pos = elecs_pos if elecs_pos is not None else []
        self.protocol = protocol if protocol is not None else []
        self.k = None

    def add_elecs_pos(self, positions):
        """
        Add multiple electrode positions.

        Parameters:
        positions (list of tuple): A list of (x, z) positions of the electrodes.
        """
        self.elecs_pos.extend(positions)

    def compute_geometrical_factor(self):
        """
        Compute the geometrical factor for a given quadrupole.

        Parameters:
        quadrupole (list of int): A list containing the indices of the electrodes for A, B, M, and N.

        Returns:
        float: The geometrical factor.
        """
        protocol = self.protocol
        elecs_pos = self.elecs_pos  # Corrected this line

        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        k = []
        for quadrupole in protocol:
            A, B, M, N = quadrupole
            r_AM = distance(elecs_pos[A], elecs_pos[M])
            r_BM = distance(elecs_pos[B], elecs_pos[M])
            r_AN = distance(elecs_pos[A], elecs_pos[N])
            r_BN = distance(elecs_pos[B], elecs_pos[N])
            K = 2 * np.pi / (1/r_AM - 1/r_BM - 1/r_AN + 1/r_BN)
            k.append(K)
        return k
    
    def add_protocol(self, protocol):
        """
        Add multiple protocol entries.

        Parameters:
        protocol (list of list): A list of quadrupoles, each containing the indices of the electrodes for A, B, M, and N.
        """
        for quadrupole in protocol:
            self.protocol.append(quadrupole)
        self.k = self.compute_geometrical_factor()  # Corrected this line

