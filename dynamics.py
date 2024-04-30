# functions used for circuit dynamics

from tableau import Tableau

def clifford_measure(tableau:Tableau, number_time_steps:int, is_odd_number_qubits:int, measurement_rate, which_pauli:list):
    # In practice, number_time_steps will be 1 to measure observables after each time, but can run for any amount of time before measuring
    # Applies two layers in 1, each applying random 1 sites, 2 sites and measurement layers
    # The difference between is that the two sites changes which qubit bonds are interacted.
    for time in range(number_time_steps):

        tableau.layer_phase_hadamard_identity()
        tableau.layer_cnots_and_identity(is_odd_number_qubits, 0, 0)
        tableau.layer_of_measurement(which_pauli, measurement_rate)

        tableau.layer_phase_hadamard_identity()
        tableau.layer_cnots_and_identity(is_odd_number_qubits, 1, 0)
        tableau.layer_of_measurement(which_pauli, measurement_rate)

def two_dimensional_clifford_measure_dynamics(tableau:Tableau, which_case:int, measurement_rate, which_pauli:list):
    # match case
    # Case 0 -> layer 0
    # Case 1 -> layer 1
    # Case 2 -> layer 2
    # Case 3 -> layer 3
    # Use modulo 4 of the time step value
    # Example: time_step = 10 -> which_case = 2

    # Each layer applies random 1 sites, 2 sites, and measurement layer

    match which_case:
        case 0:
            tableau.layer_phase_hadamard_identity()
            tableau.layer_0_2D()
            tableau.layer_of_measurement(which_pauli=which_pauli, measurement_rate=measurement_rate)
        case 1:
            tableau.layer_phase_hadamard_identity()
            tableau.layer_1_2D()
            tableau.layer_of_measurement(which_pauli=which_pauli, measurement_rate=measurement_rate)
        case 2:
            tableau.layer_phase_hadamard_identity()
            tableau.layer_2_2D()
            tableau.layer_of_measurement(which_pauli=which_pauli, measurement_rate=measurement_rate)
        case 3:
            tableau.layer_phase_hadamard_identity()
            tableau.layer_3_2D()
            tableau.layer_of_measurement(which_pauli=which_pauli, measurement_rate=measurement_rate)

