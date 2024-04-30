
import numpy as np
import copy
from utilities import g_function, count_paulis, check_commutation
import matplotlib.pyplot as plt

class Tableau:
    all = []

    def __init__(self, number_of_qubits:int, starting_state:list=[1,0]):
        # number_of_qubits: Size of system used to simulate dynamics
        # starting_state: Length 2 list denoting which Pauli operator stabilizes each qubit
        #                 -> Pauli X ---> [1, 0]
        #                 -> Pauli Z ---> [0, 1]
        #                 -> Pauli Y ---> [1, 1]

        # Verifying correct inputs
        assert number_of_qubits > 1, f"Value for number_of_qubits={number_of_qubits} was not greater than 1!"
        assert len(starting_state) == 2, "Value for starting_state should be equal to 2!"
        assert type(starting_state[0]) == int and type(starting_state[1]) == int, "Values inside starting_state must be integers!"
        assert starting_state[0] == 0 or starting_state[0] == 1, "Values inside starting_state must be either 0 or 1!"
        assert starting_state[1] == 0 or starting_state[1] == 1, "Values inside starting_state must be either 0 or 1!"
        assert not np.array_equiv(starting_state, [0,0]), "Both indices of starting_state must not both be 0!"

        # Creating values for the class
        self.number_of_qubits = number_of_qubits
        self.generator_array = np.zeros(  # Creates generator array that tracks stabilizer dynamics 
            shape=(number_of_qubits, 2 * number_of_qubits),
            dtype=int
            )
        
        # Creates sign array to be able to track exact sign changes in stabilizers
        # 0 -> Positive,   1 -> Negative
        self.sign_array = np.zeros(  
            shape=(number_of_qubits, 1),
            dtype=int
            )
        
        # Applying starting_state value to generator_array.
        # Each qubit has a generator with the starting state pauli located at its position.
        for position in range(number_of_qubits):
            self.generator_array[position, 2 * position] = starting_state[0]
            self.generator_array[position, 2 * position + 1] = starting_state[1]

    @classmethod
    def initialization_from_csv(cls, filename):
        # Initializes from a csv files
        # CSV file should contain an n x 2n+1 array
        # n is the number of qubits
        # Extra column comes from the sign array
        # Does not automatically skip a row in case of a header
        stored_array = np.loadtxt(filename, dtype=int, delimiter=",", skiprows=0)

        # Creating a blank Tableau object
        tab_object = Tableau(np.shape(stored_array)[0])

        # Assigning values to new object
        tab_object.generator_array = stored_array[:,:-1]
        for row in range(len(stored_array[:,-1])):
            tab_object.sign_array[row,0] = stored_array[row,-1]
        
        del stored_array
        return tab_object
    
    def save_to_csv(self, output_filename:str, header:str=""):
        # Saves tableau data to CSV file
        # Concatenates generator and sign arrays and stores in the CSV
        # Input desired file name and optional header
        store_array = np.concatenate([self.generator_array, self.sign_array], axis=1)
        np.savetxt(output_filename, store_array, fmt="%1.0f", delimiter=",", header=header)
        pass

    def __repr__(self):
        # Allows the generator array to be printed when using print(tableau_object)
        print(self.generator_array)
        return ""
    
    # Defining unitary gate functions to apply to the tableaus for dynamics

    def controlled_not_left_gate(self, control_position:int):
        # Control position ranges from 0 to (number of qubits - 1)
        # Periodic coordinates allowed
        # Target qubit is located 1 step to the right
        #       ex. control_position = 2, target_position = 3
        # Algorithm:
        # Varible a labels control, b lables target, r lables sign array value
        # r = r ⊕ xazb(xb ⊕ za ⊕ 1)
        # xb = xa ⊕ xb, za = za ⊕ zb
        assert 0 <= control_position < self.number_of_qubits, f"Value control_position = {control_position} is not within system size."
        tableau_shape = np.shape(self.generator_array)
        
        if control_position != self.number_of_qubits-1:
            for row in range(tableau_shape[0]):
                x_a, z_a, x_b, z_b = self.generator_array[row, 2*control_position:2*control_position + 4]
                self.sign_array[row,0] = (self.sign_array[row,0] + x_a*z_b*((x_b+z_a+1) % 2)) % 2
                self.generator_array[row, 2*control_position+2] = (x_a + x_b) % 2
                self.generator_array[row, 2*control_position+1] = (z_a + z_b) % 2
        else:
            for row in range(tableau_shape[0]):
                x_a, z_a, x_b, z_b = self.generator_array[row, [-2,-1,0,1]]
                self.sign_array[row,0] = (self.sign_array[row,0] + x_a*z_b*((x_b+z_a+1) % 2)) % 2
                self.generator_array[row, 0] = (x_a + x_b) % 2
                self.generator_array[row, -1] = (z_a + z_b) % 2

    def controlled_not_right_gate(self, control_position:int):
        # Control position ranges from 0 to (number of qubits - 1)
        # Periodic coordinates allowed
        # Target qubit is located 1 step to the left
        #       ex. control_position = 2, target_position = 1
        # Algorithm:
        # Varible a labels control, b lables target, r lables sign array value
        # r = r ⊕ xazb(xb ⊕ za ⊕ 1)
        # xb = xa ⊕ xb, za = za ⊕ zb
        assert 0 <= control_position < self.number_of_qubits, f"Value control_position = {control_position} is not within system size."
        tableau_shape = np.shape(self.generator_array)

        if control_position != 0:
            for row in range(tableau_shape[0]):
                x_b, z_b, x_a, z_a = self.generator_array[row, 2*control_position-2:2*control_position+2]
                self.sign_array[row,0] = (self.sign_array[row,0] + x_a*z_b*((x_b+z_a+1) % 2)) % 2
                self.generator_array[row, 2*control_position-2] = (x_a + x_b) % 2
                self.generator_array[row, 2*control_position+1] = (z_a + z_b) % 2
        else:
            for row in range(tableau_shape[0]):
                x_b, z_b, x_a, z_a = self.generator_array[row, [-2,-1,0,1]]
                self.sign_array[row,0] = (self.sign_array[row,0] + x_a*z_b*((x_b+z_a+1) % 2)) % 2
                self.generator_array[row, -2] = (x_a + x_b) % 2
                self.generator_array[row, 1] = (z_a + z_b) % 2        

    def nonlocal_controlled_not_gate(self, control_position:int, target_position:int):
        # Can apply CNOT gate to any two qubits, no longer need to be next to each other
        # Control indicies labelled with _c
        # Target indicies labelled with _t
        tableau_shape = np.shape(self.generator_array)

        for row in range(tableau_shape[0]):
            x_c, z_c = self.generator_array[row, 2*control_position:2*control_position+2]
            x_t, z_t = self.generator_array[row, 2*target_position:2*target_position+2]
            self.sign_array[row,0] = (self.sign_array[row,0] + x_c*z_t*((x_t+z_c+1) % 2)) % 2
            self.generator_array[row, 2*target_position] = (x_c + x_t) % 2
            self.generator_array[row, 2*control_position+1] = (z_c + z_t) % 2

    def phase_gate(self, position:int):
        # Applies a phase unitary gate
        # Position ranges from 0 to (number_of_qubits - 1)
        # Algorithm:
        # r = r ⊕ xizi
        # zi = zi ⊕ xi

        assert 0 <= position < self.number_of_qubits, f"Value position = {position} is not within system size."
        tableau_shape = np.shape(self.generator_array)

        for row in range(tableau_shape[0]):
            x_i, z_i = self.generator_array[row, 2*position:2*position+2]
            self.sign_array[row, 0] = (self.sign_array[row, 0] + x_i*z_i) % 2
            self.generator_array[row, 2*position+1] = (z_i + x_i) % 2
        
    def hadamard_gate(self, position:int):
        # Applies a Hadamard unitary gate
        # Position ranges from 0 to (number_of_qubits - 1)
        # Algorithm:
        # r = r ⊕ xizi
        # Swap x_i and z_i

        assert 0 <= position < self.number_of_qubits, f"Value position = {position} is not within system size."
        tableau_shape = np.shape(self.generator_array)
        
        for row in range(tableau_shape[0]):
            x_i, z_i = self.generator_array[row, 2*position:2*position+2]
            self.sign_array[row, 0] = (self.sign_array[row, 0] + x_i*z_i) % 2
            self.generator_array[row, 2*position] = z_i
            self.generator_array[row, 2*position+1] = x_i

    def random_cnots_identity(self, position:int):
        # Position ranges from 0 to (number of qubits - 1)
        # Periodic coordinates allowed
        # Randomly chooses between two CNOT gates and the identity(does nothing)
        # This creates a simplified version of random Clifford dynamics

        # Assertion checks
        assert 0 <= position < self.number_of_qubits, f"Value position = {position} is not within system size."

        # Applying random gate of three choices
        random_var = np.random.rand()
        if random_var < 1/3:
            self.controlled_not_left_gate(position)
        elif random_var < 2/3:
            # In practice, the positions will be placed at every two qubits.
            # The CNOT Right gate has an offset on the control qubit, which must be accounted for.
            self.controlled_not_right_gate(position + 1)
        # Note the identity does not need an explicit call
    
    def random_phase_hadamard_identity(self, position:int):
        # Position ranges from 0 to (number of qubits - 1)
        # Periodic coordinates allowed
        # Randomly chooses between phase, hadamard, and the identity(does nothing)
        # This creates a version of random Clifford dynamics

        # Assertion checks
        assert 0 <= position < self.number_of_qubits, f"Value position = {position} is not within system size."

        # Applying random gate of three choices
        random_var = np.random.rand()
        if random_var < 1/3:
            self.phase_gate(position)
        elif random_var < 2/3:
            self.hadamard_gate(position)
        # Note the identity does not need an explicit call\
         
    def random_nonlocal_cnot_identity(self, position1:int, position2:int):
        # Applying random gate of three choices
        random_var = np.random.rand()
        if random_var < 1/3:
            self.nonlocal_controlled_not_gate(position1, position2)
        elif random_var < 2/3:
            self.nonlocal_controlled_not_gate(position2, position1)
        else:
            pass
        # Note the identity does not need an explicit call

    # Defining methods to manipulate the tableau and extract information
            
    def swap_rows(self, row_a:int, row_b:int):
        # Method to swap two rows in the generator tableau and sign array
        # row_a ranges from 0 to (number_of_qubits - 1)
        # row_b ranges from 0 to (number_of_qubits - 1)

        # Assertion checks
        assert 0 <= row_a < self.number_of_qubits, f"Value row_a = {row_a} is not within system size."
        assert 0 <= row_b < self.number_of_qubits, f"Value row_b = {row_b} is not within system size."

        # Assigning dummy values for to correctly overwrite
        dummy_row_a = self.generator_array[row_a, :].copy()
        dummy_row_b = self.generator_array[row_b, :].copy()
        dummy_sign_a = self.sign_array[row_a, 0]

        # Swapping rows in generator array and sign array
        self.generator_array[row_a, :] = dummy_row_b
        self.generator_array[row_b, :] = dummy_row_a
        self.sign_array[row_a, 0] = self.sign_array[row_b, 0]
        self.sign_array[row_b, 0] = dummy_sign_a

    def row_sum(self, row_a: int, row_b: int):
        # Multiplies row_b by row_a and replace row_b by the product
        # row_a, row_b ranges from 0 to (number_of_qubits - 1)

        # Assertion checks
        assert 0 <= row_a < self.number_of_qubits, f"Value row_a = {row_a} is not within system size."
        assert 0 <= row_b < self.number_of_qubits, f"Value row_b = {row_b} is not within system size."

        # Creating dummy arrays
        copy_row_a = self.generator_array[row_a, :]
        copy_row_b = self.generator_array[row_b, :]
        dummy_var = 0

        # Summing row values
        self.generator_array[row_b, :] = (copy_row_a + copy_row_b) % 2

        # Sign array values calculated with algorithm using g_function in utilities
        for position in range(int(self.number_of_qubits/2)):
            dummy_var += g_function(copy_row_a[2*position], copy_row_a[2*position + 1], copy_row_b[2*position], copy_row_b[2*position + 1])
        self.sign_array[row_b, 0] = (self.sign_array[row_a, 0] + self.sign_array[row_b, 0] + int((dummy_var % 4)/2)) % 2

    def sort_active_region(self, active_row: int, active_col: int):
        # Sorts matrix in active column based on pauli matricies
        # Sorted order is X, Z, Y, I or (1,2,3,4)
        # active_row, active_col range from 0 to (number_of_qubits - 1)

        # Assertion checks
        assert 0 <= active_row < self.number_of_qubits, f"Value active_row = {active_row} is out of bounds of generator array."
        assert 0 <= active_col < self.number_of_qubits, f"Value active_col = {active_col} is out of bounds of generator array."

        x, y, z = [1, 0], [1, 1], [0, 1]
        location_of_paulis = np.zeros(shape = np.shape(self.generator_array)[0], dtype = int)

        # List which pauli is in which row in location_of_paulis list
        for row in range(self.number_of_qubits):
            pauli = [self.generator_array[row, 2*active_col], self.generator_array[row, 2*active_col+1]]
            if pauli == x:
                location_of_paulis[row] = 1
            elif pauli == z:
                location_of_paulis[row] = 2
            elif pauli == y:
                location_of_paulis[row] = 3
            else:
                location_of_paulis[row] = 4

        # Sorting location_of_paulis in new list
        sorted_location = np.sort(location_of_paulis[active_row:])

        # Actual sorting of generator array
        for element in range(len(sorted_location)):
            position = element + active_row
            while sorted_location[element] != location_of_paulis[position]:
                position += 1
            a = location_of_paulis[element + active_row]
            b = location_of_paulis[position]
            location_of_paulis[element + active_row] = b
            location_of_paulis[position] = a
            self.swap_rows(element + active_row, position)

        return location_of_paulis

    def transform_to_rref(self):
        # Transforms generator array in Row-Reduced Echelon Form
        # Should not change state of the object, so input is specific array.

        shape = np.shape(self.generator_array)
        active_col = 0
        active_row = 0
        while active_row < shape[0] and active_col < int((shape[1])/2):
            pauli_location = self.sort_active_region(active_row, active_col)
            pauli_count = count_paulis(pauli_location, active_row)
            
            if pauli_count == 0:
                active_col += 1

            elif pauli_count == 1:
                for element in range(len(pauli_location)):
                    if element > active_row and pauli_location[active_row] == pauli_location[element]:
                        self.row_sum(active_row, element)
                dummy_positions = self.sort_active_region(active_row, active_col)
                active_row += 1
                active_col += 1
            
            elif pauli_count == 2:
                for element in range(len(pauli_location)):
                    if element > active_row and pauli_location[active_row] == pauli_location[element]:
                        self.row_sum(active_row, element)
                    elif element > active_row and pauli_location[active_row] != pauli_location[element]:
                        if pauli_location[element] != pauli_location[element - 1] and pauli_location[element] != 4:
                            new_position = element
                        if element > new_position and pauli_location[element] != 4:
                            self.row_sum(new_position, element)
                dummy_positions = self.sort_active_region(active_row, active_col)
                active_row += 2
                active_col += 1

            elif pauli_count == 3:

                second_location = 0
                third_location = 0

                for element in range(len(pauli_location)):
                    if element > active_row and pauli_location[element] != 4:
                        if second_location == 0 and (pauli_location[element] != pauli_location[element - 1]):
                            second_location = element
                        elif second_location != 0 and (pauli_location[element] != pauli_location[element - 1]):
                            third_location = element

                for element in range(len(pauli_location)):
                    if element > active_row and pauli_location[element] != 4:
                        if pauli_location[element] == 1 and (element != active_row):
                            self.row_sum(active_row, element)
                        elif pauli_location[element] == 2 and element != second_location:
                            self.row_sum(second_location, element)
                        elif pauli_location[element] == 3 and element != third_location:
                            self.row_sum(third_location, element)
            
                dummy_positions = self.sort_active_region(active_row, active_col)
                self.row_sum(active_row, active_row + 2)
                self.row_sum(active_row + 1, active_row + 2)
                active_row += 2
                active_col += 1

    def get_rank_tableau(self, system_end:int, system_start:int=0):
        # Tableau must be row reduced
        # essentially gives the number of linearly independent generators within the system chosen

        number_zero_rows = 0
        for row in range(self.number_of_qubits):
            # Truncates tableau by only looking up to system_end
            if len(set(self.generator_array[row, :2*(system_end-system_start+1)])) == 1 and self.generator_array[row, 0] == 0:
                number_zero_rows += 1   
        return self.number_of_qubits - number_zero_rows    

    def get_entanglement_entropy(self, system_end:int, system_start:int = 0):
        # Uses dummy tableau objects to avoid altering the state of the system per calculation
        # Although theoretically, the row operations should not affect the state
        # Calculates entanglement entropy of current state of a subsystem
        # Subsystem can be any contiguous region
        # Entropy = Rank - System_Size

        # Assertion Checks
        assert 0 <= system_start < self.number_of_qubits, f"Value system_start = {system_start} is not within system bounds."
        assert 0 <= system_end < self.number_of_qubits, f"Value system_end = {system_end} is not within system bounds."
        assert system_end >= system_start, f"Value system_start={system_start} must be less than system_end={system_end}"

        dummy_tableau = copy.deepcopy(self)
        # Rearranging generator array to isolate system
        if system_start == 0:
            dummy_tableau.transform_to_rref()
        else:
            dummy_tableau.generator_array = dummy_tableau.generator_array[:,list(range(2*system_start, 2*(system_end+1)))+list(range(0, 2*system_start))+list(range(2*(system_end+1), 2*self.number_of_qubits))]
            dummy_tableau.transform_to_rref()
        
        # Finding rank of the generator_array
        rank = dummy_tableau.get_rank_tableau(system_end, system_start=system_start)
        
        # Finding entanglement entropy
        entropy = np.abs(rank) - np.abs(system_end - system_start + 1)

        return entropy
    
    def get_entanglement_entropy_for_separated_systems(self, system_A:list, system_B:list):
        # Uses dummy tableau objects to avoid altering the state of the system per calculation
        # Although theoretically, the row operations should not affect the state
        # This is a function designed to locate the columns of two specific ranges
        # of qubits and computing the entanglement entropy as one system.
        # Reorders copy of the stabilizer tableau to calculate this observable.
        # Both systems are relocated to left of tableau before proceeding.

        # Creates copy of the Tableau object
        dummy_tableau = copy.deepcopy(self)

        # Reorders columns based on the different lists of systems
        dummy_tableau.generator_array = dummy_tableau.generator_array[:,list(range(2*system_A[0], 2*(system_A[1]+1)))+list(range(2*system_B[0], 2*(system_B[1]+1)))+list(range(0, 2*system_A[0]))+list(range(2*(system_A[1]+1), 2*system_B[0]))+list(range(2*(system_B[1]+1), 2*self.number_of_qubits))]

        # Calculates entropy and return value
        return dummy_tableau.get_entanglement_entropy(system_A[1] + system_B[1] - system_A[0] - system_B[0] + 1,0)  
    
    def get_entanglement_entropy_for_three_separated_systems(self, system_A:list, system_B:list, system_C:list):
        # Uses dummy tableau objects to avoid altering the state of the system per calculation
        # Although theoretically, the row operations should not affect the state
        # This is a function designed to locate the columns of three specific ranges
        # of qubits and computing the entanglement entropy as one system.
        # Reorders copy of the stabilizer tableau to calculate this observable.
        # All three subsystems relocated to left of tableau before proceeding.
  
        # Creates copy of the Tableau object
        dummy_tableau = copy.deepcopy(self)

        # Reorders columns based on the different lists of systems
        dummy_tableau.generator_array = dummy_tableau.generator_array[:,list(range(2*system_A[0], 2*(system_A[1]+1)))+list(range(2*system_B[0], 2*(system_B[1]+1)))+list(range(2*system_C[0], 2*(system_C[1]+1)))+list(range(0, 2*system_A[0]))+list(range(2*(system_A[1]+1), 2*system_B[0]))+list(range(2*(system_B[1]+1), 2*system_C[0]))+list(range(2*(system_C[1]+1), 2*self.number_of_qubits))]

        # Calculates entropy and return value
        return dummy_tableau.get_entanglement_entropy(system_A[1] + system_B[1] + system_C[1] - system_A[0] - system_B[0] - system_C[0] + 2,0)

    def get_entanglement_negativity(self, system_A:list, system_B:list):
        # Uses dummy tableau objects to avoid altering the state of the system per calculation
        # This function calculates the entanglement negativity between system_A and system_B
        # The parameters should be a 2 entry list tuple, where the endpoints of the system are the indexed values qubit positions
        # EX 4 qubits have positions 0,1,2,3
        # The algorithm finds the generators thats are only contained within both A and B
        # then uses the parts of these generators only contained within B to create a matrix containing their commutation relation values.
        # 1/2 x Rank of this Matrix is the Entanglement Negativity

        # Create dummy tableau
        dummy_tableau = copy.deepcopy(self)

        # Change column order to complement(AB) A  B
        dummy_tableau.generator_array = dummy_tableau.generator_array[:,list(range(0, 2*system_A[0]))+list(range(2*(system_A[1]+1), 2*system_B[0]))+list(range(2*(system_B[1]+1), 2*self.number_of_qubits))+list(range(2*system_A[0], 2*(system_A[1]+1)))+list(range(2*system_B[0], 2*(system_B[1]+1)))]
        
        # Transform tableau to RREF
        dummy_tableau.transform_to_rref()

        # Find last qubit position in complement of AB
        # minus 3 due to length of two lists plus indexing of matrix to get qubit label
        endpoint_comp_AB = self.number_of_qubits - system_A[1] - system_B[1] + system_A[0] + system_B[0] - 3

        # Find the number of rows that are nonzero and not within complement(AB)
        number_nonzero_generators_AB = 0
        generator_positions = np.array([],dtype=int)
        if endpoint_comp_AB != -1:
            for generator in range(self.number_of_qubits):
                if len(set(dummy_tableau.generator_array[generator, :2*(endpoint_comp_AB+1)])) == 1 and dummy_tableau.generator_array[generator, 0] == 0:
                    if len(set(dummy_tableau.generator_array[generator,:])) == 2:
                        number_nonzero_generators_AB += 1
                        generator_positions = np.concatenate([generator_positions, [generator]], axis = 0, dtype=int)
        else: # Added contingency for complement of AB endpoint being negative one which fails the first set condition
            for generator in range(self.number_of_qubits):
                if len(set(dummy_tableau.generator_array[generator,:])) == 2:
                    number_nonzero_generators_AB += 1
                    generator_positions = np.concatenate([generator_positions, [generator]], axis = 0, dtype=int)           

        # Create matrix with generators only contained within B
        truncated_matrix = dummy_tableau.generator_array[generator_positions,2*(self.number_of_qubits-system_B[1]+system_B[0]-1):]

        # Create commutation J matrix by testing commutation relation between each combination of generators
        J_matrix = np.zeros((number_nonzero_generators_AB, number_nonzero_generators_AB), dtype=int)
        for row1 in range(number_nonzero_generators_AB):
            for row2 in range(number_nonzero_generators_AB):
                J_matrix[row1,row2] = check_commutation(truncated_matrix[row1,:], truncated_matrix[row2,:])
        if np.shape(J_matrix) == (0,0):
            return 0

        # Put J matrix in tableau object to transform to RREF and compute rank
        if np.shape(J_matrix)[0]%2 == 1:
            J_matrix = np.concatenate([J_matrix, np.zeros((1,np.shape(J_matrix)[1]), dtype = int)], axis=0)
            J_matrix = np.concatenate([J_matrix, np.zeros((np.shape(J_matrix)[0],1),dtype=int)],axis=1)
        J_tableau = Tableau(np.shape(J_matrix)[0])
        J_tableau.generator_array = J_matrix
        J_tableau.transform_to_rref()

        rank = J_tableau.get_rank_tableau(np.shape(J_matrix)[0]-1)

        # return 1 half of rank to get entanglement negativity
        return int(rank/2)

    def get_bipartite_mutual_information(self, system_A:list, system_B:list):
        # Function to calculate the mutual information between system_A and system_B
        # Interpreted as the quantum information shared by the systems
        # Formula: I = S_A + S_B - S_AB

        # Calculating entropy for separate systems
        entropy_A = self.get_entanglement_entropy(system_A[1], system_A[0])
        entropy_B = self.get_entanglement_entropy(system_B[1], system_B[0])

        # Calculating entropy for the two combined systems
        entropy_AB = self.get_entanglement_entropy_for_separated_systems(system_A, system_B)

        return entropy_A + entropy_B - entropy_AB
    
    def get_tripartite_mutual_information(self, system_A:list, system_B:list, system_C:list):
        # same as bipartite MI, but finds shared information between 3 systems
        # I = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC

        # Getting entropy for single systems
        entropy_A = self.get_entanglement_entropy(system_A[1], system_A[0])
        entropy_B = self.get_entanglement_entropy(system_B[1], system_B[0])
        entropy_C = self.get_entanglement_entropy(system_C[1], system_C[0])

        # Getting entropy for two systems
        entropy_AB = self.get_entanglement_entropy_for_separated_systems(system_A, system_B)
        entropy_BC = self.get_entanglement_entropy_for_separated_systems(system_B, system_C)
        entropy_AC = self.get_entanglement_entropy_for_separated_systems(system_A, system_C)

        # Getting final entropy for three combined systems
        entropy_ABC = self.get_entanglement_entropy_for_three_separated_systems(system_A, system_B, system_C)

        return entropy_A + entropy_B + entropy_C - entropy_AB - entropy_AC - entropy_BC + entropy_ABC

    # Defining methods to measure qubits in the system and update tableau accordingly

    def measure_sites(self, sites:list, measurement:list = [1,0]):
        # sites: qubit positions for measurement to take place at one time, all have to be same measurement
        # measurement: choose which Cartesian direction to measure the state of the qubit
        # X = [1,0], Y = [1,1], Z = [1,1]
        # IMPORTANT: if using only CNOT gates for entanglement dynamics, the measurement must be Y, other wise the entanglement will always decay at long time

        # Algorithm checks each generator if it commutes with the measurement Pauli String, ergo measurement tuple placed at each measurement site in a 2*number_qubits length row array
        # If any generators anticommute, they will be rowsummed with the first anticommuting generator
        # Then the first anticommuting generator is replaced by the measurement Pauli String
        # The measurement outcome is tracked by randomly replacing the corresponding sign_array value in the place of the first anticommuting generator with a 1 or 0
        # Essentially if spin up or down
    
        #print('Measure Array:\n', measure_array)
        shape = np.shape(self.generator_array)
        first_anticommuting_pos = 0

        commutation_locations = np.zeros(shape[0], dtype = int)
        for row in range(shape[0]):
            dummy = 0
            for position in sites:
                dummy += self.generator_array[row,2*position] * measurement[1] + measurement[0] * self.generator_array[row,2 * position + 1]
            commutation_locations[row] = dummy % 2    


        #print('Locations of anticommutors: \n', commutation_locations)

    
        for element in range(len(commutation_locations)):
            if first_anticommuting_pos == 0 and commutation_locations[element] == 1:
                first_anticommuting_pos = element
                #print('first locations is', first_anticommuting_pos)
                break
    
        for element in range(len(commutation_locations)):
            if element > first_anticommuting_pos and commutation_locations[element] == 1:
                self.row_sum(first_anticommuting_pos, element)
                #print(f'loop: {element}\n', tableau)

        if commutation_locations[first_anticommuting_pos] == 1:
            self.generator_array[first_anticommuting_pos,:] = np.zeros(shape=shape[1])
            for position in sites:
                self.generator_array[first_anticommuting_pos, 2*position:2*position+2] = measurement
            #print('Swaping out M\n', tableau)
            random_int = np.random.randint(0,2)
            self.sign_array[first_anticommuting_pos, 0] = random_int
            #print('adding random int: \n', tableau)

    # Defining methods to apply layers of gates and measurements
    
    def layer_cnots_and_identity(self, is_odd_number_qubits:int, is_applied_odd_bonds:int, periodic_boundary:int):
        # is_applied_odd_bonds: True if the two site unitary is first applied between qubits 1 and 2, not 0 and 1
        # is_odd_number_qubits: Total system number is odd
        # periodic_boundary: whether or not qubits on the ends of the system can interact
        # All three of these arguments are 1 if true, 0 if false
        # This functions applies a full layer of the random cnot gates and identities in a 1 D System
        # Skips every two qubits, but how the gates are applied is controlled by the arguments

        # Assertion Checks
        assert is_odd_number_qubits == 1 or is_odd_number_qubits == 0, f"is_odd_number_qubits should be a 1 or 0."
        assert is_applied_odd_bonds == 1 or is_applied_odd_bonds == 0, f"is_applied_odd_bonds should be a 1 or 0."
        assert periodic_boundary == 1 or periodic_boundary == 0, f"periodic_boundary should be a 1 or 0."
        assert self.number_of_qubits != 1, f"Two site unitaries require at least two sites!"

        # Applying layer
        shape = np.shape(self.generator_array)
        if is_odd_number_qubits == 0:
            # Even number of qubits
            if is_applied_odd_bonds == 1:
                # Odd bonds make period boundary important to extend range of indicies
                for gateIndex in range(int(shape[0]/2 - 1 + periodic_boundary)):
                    self.random_cnots_identity(2 *gateIndex + 1)
            elif shape[0] == 2:
                # For two qubits, just let them interact as if with even bonds
                for gateIndex in range(int(shape[0]/2)):
                    self.random_cnots_identity(2 * gateIndex)
            else:
                # Even bonds not equal to 2
                for gateIndex in range(int(shape[0]/2)):
                    self.random_cnots_identity(2 * gateIndex)
        elif is_odd_number_qubits == 1:
            # Periodic boundary does not make sense with an odd number of qubits since 
            # one qubit would be interacted with twice in one layer
            if is_applied_odd_bonds == 1:
                for gateIndex in range(int((shape[0]-1)/2)):
                    self.random_cnots_identity(2 * gateIndex + 1)
            else:
                for gateIndex in range(int((shape[0]-1)/2)):
                    self.random_cnots_identity(2 * gateIndex)

    def layer_phase_hadamard_identity(self):
        # For a 1-D or 2-D since these gates are 1 site unitaries
        # Applies a random gate to each qubit of the three available

        for gateIndex in range(np.shape(self.generator_array)[0]):
            self.random_phase_hadamard_identity(gateIndex)

    def layer_of_measurement(self, which_pauli:list, measurement_rate = 0.0):
        # For a 1-D or 2-D since these gates are 1 site measurements
        # Applies a measurment randomly to each qubit based on the measurement_rate

        shape = np.shape(self.generator_array)

        for position in range(shape[0]):
            random_num = np.random.rand()
            if random_num < measurement_rate:
                # print(position, random_num)
                self.measure_sites([position], measurement=which_pauli)

    # Defining methods to apply layers of gates and measurements to 2D system

    # Note 2-D Systems use the 1-D system Implementation but applying gates requires care
    # Nonlocal CNOTS are used to skip onto new rows
    # Arrangement of 2-D system of 16 qubits:
    #  0 X X X X     The qubits are still labeled 0 to 15
    #  4 X X X X     The qubit starting the row is listed before each row
    #  8 X X X X     This shows why nonlocal gates are needed to interact, say, qubit 5 and qubit 9
    # 12 X X X X     Physically, they should be next to each other

    def row_layer(self, row:int, is_odd_side_length:int, is_applied_odd_bonds:int):
        # Applies random layer of CNOT-R and CNOT-L and Identity to specific row of 2D qubit system
        # row: listing which row this layer is applied to; 0 to sqrt(number_of_qubits)-1
        # is_odd_side_length: 1 if sqrt(num_qubits) is odd, 0 if even
        # is_applied_odd_bonds: 0 if first qubit in row is input into gate, 1 if not

        # Assertion checks
        assert self.number_of_qubits > 1, "System must have more than one body for interaction"

        # Applying layer
        number_qubits_per_row = int(np.sqrt(self.number_of_qubits))

        # skipped_position is the counting of gates along a row
        # Each gate happens every two qubits, so actual position along row is
        # 2*skipped_position + number_qubits_per_row*row
        if is_odd_side_length == 0:
            if is_applied_odd_bonds == 0:
                for skipped_position in range(int(number_qubits_per_row/2)):
                    self.random_nonlocal_cnot_identity(2*skipped_position + row*number_qubits_per_row, 2*skipped_position+1 + row*number_qubits_per_row)
            elif is_applied_odd_bonds == 1:
                if number_qubits_per_row == 2:
                    self.random_nonlocal_cnot_identity(2*row, 2*row+1)
                else:
                    for skipped_position in range(int(number_qubits_per_row/2)-1):
                        self.random_nonlocal_cnot_identity(2*skipped_position+1 + row*number_qubits_per_row, 2*skipped_position+2 + row*number_qubits_per_row)
        elif is_odd_side_length == 1:
            if is_applied_odd_bonds == 0:
                for skipped_position in range(int((number_qubits_per_row-1)/2)):
                    self.random_nonlocal_cnot_identity(2*skipped_position + row*number_qubits_per_row, 2*skipped_position+1 + row*number_qubits_per_row)
            elif is_applied_odd_bonds == 1:
                for skipped_position in range(int((number_qubits_per_row-1)/2)):
                    self.random_nonlocal_cnot_identity(2*skipped_position+1 + row*number_qubits_per_row, 2*skipped_position+2 + row*number_qubits_per_row)

    def column_layer(self, column:int, is_odd_side_length:int, is_applied_odd_bonds:int):
        # Applies random layer of CNOT-R and CNOT-L and Identity to specific column of 2D qubit system
        # column: listing which column this layer is applied to; 0 to sqrt(number_of_qubits)-1
        # is_odd_side_length: 1 if sqrt(num_qubits) is odd, 0 if even
        # is_applied_odd_bonds: 0 if first qubit in row is input into gate, 1 if not

        # Assertion checks
        assert self.number_of_qubits > 1, "System must have more than one body for interaction"

        # Applying layer
        number_qubits_per_col = int(np.sqrt(self.number_of_qubits))

        # skipped_position is the counting of gates along a row
        # Each gate happens every two qubits, so actual position along row is
        # 2*skipped_position + number_qubits_per_row*row
        if is_odd_side_length == 0:
            if is_applied_odd_bonds == 0:
                for skipped_position in range(int(number_qubits_per_col/2)):
                    self.random_nonlocal_cnot_identity(column + 2*number_qubits_per_col*skipped_position, column + number_qubits_per_col*(2*skipped_position+1))
            elif is_applied_odd_bonds == 1:
                if number_qubits_per_col == 2:
                    self.random_nonlocal_cnot_identity(column, column + 2)
                else:
                    for skipped_position in range(int(number_qubits_per_col/2)-1):
                        self.random_nonlocal_cnot_identity(column + number_qubits_per_col*(2*skipped_position+1), column + number_qubits_per_col*(2*skipped_position+2))
        elif is_odd_side_length == 1:
            if is_applied_odd_bonds == 0:
                for skipped_position in range(int((number_qubits_per_col-1)/2)):
                    self.random_nonlocal_cnot_identity(column + 2*number_qubits_per_col*skipped_position, column + number_qubits_per_col*(2*skipped_position+1))
            elif is_applied_odd_bonds == 1:
                for skipped_position in range(int((number_qubits_per_col-1)/2)):
                    self.random_nonlocal_cnot_identity(column + number_qubits_per_col*(2*skipped_position+1), column + number_qubits_per_col*(2*skipped_position+2))

    # The 2-D layers of 2 site unitaries are broken up into four types:
    # Layer_0: Row layers where 0th row is applied even bonds and alternates each row
    # Layer_1: Column layers where 0th column is applied odd bonds and alternates each row
    # Layer_2: Row layers where 0th row is applied odd bonds and alternates each row
    # Layer_3: Column layers where 0th column is applied even bonds and alternates each row
    # This ensures that the system interacts uniformly as time increases
    # When using time for loop, time mod 4 can be used to apply each type of layer using match/case or if statements
    def layer_0_2D(self):
        for row in range(int(np.sqrt(self.number_of_qubits))):
            self.row_layer(row, np.sqrt(self.number_of_qubits)%2, row%2)

    def layer_1_2D(self):
        for column in range(int(np.sqrt(self.number_of_qubits))):
            self.column_layer(column, np.sqrt(self.number_of_qubits)%2, (column+1)%2)

    def layer_2_2D(self):
        for row in range(int(np.sqrt(self.number_of_qubits))):
            self.row_layer(row, np.sqrt(self.number_of_qubits)%2, (row+1)%2)

    def layer_3_2D(self):
        for column in range(int(np.sqrt(self.number_of_qubits))):
            self.column_layer(column, np.sqrt(self.number_of_qubits)%2, (column)%2)
