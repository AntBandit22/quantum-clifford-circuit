# Example File to run data

from tableau import Tableau
import numpy as np
import time
from datetime import date

NUM_QUBITS = 16
SYSTEM_DIMENSION = 2
INITIAL_STATE = [1,1]
ODD_QUBITS = NUM_QUBITS % 2
NUM_TIME_STEPS = 100
WHICH_PAULI = [1,1] # What type of measurement
NUM_TRIALS = 30 # How many per file
RATE_ARRAY = np.linspace(0,0.50,51)
TODAY = date.today()

# For the trivial system division
SYSTEM_A = [0,int((NUM_QUBITS/4)-1)]
SYSTEM_B = [int(NUM_QUBITS/4), int(NUM_QUBITS/2-1)]
SYSTEM_C = [int(NUM_QUBITS/2), int((3*NUM_QUBITS/4)-1)]
SYSTEM_D = [int(3*NUM_QUBITS/4), NUM_QUBITS - 1]

# Bipartition systems
SYSTEM_1 = [0,int(NUM_QUBITS/2-1)]
SYSTEM_2 = [int(NUM_QUBITS/2), NUM_QUBITS-1]


for rate in range(len(RATE_ARRAY)):
    # Creating arrays that will be exported as final data sets
    eeB_export_data = np.zeros((NUM_TIME_STEPS, 1))
    eeAC_export_data = np.zeros((NUM_TIME_STEPS, 1))
    eeBC_export_data = np.zeros((NUM_TIME_STEPS, 1))
    miAC_export_data = np.zeros((NUM_TIME_STEPS, 1)) 
    miAD_export_data = np.zeros((NUM_TIME_STEPS, 1))
    miBC_export_data = np.zeros((NUM_TIME_STEPS, 1))
    enAC_export_data = np.zeros((NUM_TIME_STEPS, 1))
    enAD_export_data = np.zeros((NUM_TIME_STEPS, 1))
    enBC_export_data = np.zeros((NUM_TIME_STEPS, 1))

    for trial in range(NUM_TRIALS):

        tab = Tableau(NUM_QUBITS, starting_state=INITIAL_STATE)
        
        # Creating data sets for each measurement trial with time value and observable value
        eeB_data_set = np.zeros((NUM_TIME_STEPS, 2))
        eeAC_data_set = np.zeros((NUM_TIME_STEPS, 2))
        eeBC_data_set = np.zeros((NUM_TIME_STEPS, 2))
        miAC_data_set = np.zeros((NUM_TIME_STEPS, 2)) 
        miAD_data_set = np.zeros((NUM_TIME_STEPS, 2))
        miBC_data_set = np.zeros((NUM_TIME_STEPS, 2))
        enAC_data_set = np.zeros((NUM_TIME_STEPS, 2))
        enAD_data_set = np.zeros((NUM_TIME_STEPS, 2))
        enBC_data_set = np.zeros((NUM_TIME_STEPS, 2))

        toc = time.perf_counter()

        for time_step in range(NUM_TIME_STEPS):
            # Performing dynamics of time step
            two_dimensional_clifford_measure_dynamics(tab, which_case=time_step%4, measurement_rate=RATE_ARRAY[rate], which_pauli=WHICH_PAULI)

            # Measuring Entanglement Entropy and Negativity, and Mutual Information for this time step after dynamics
            entropyB = tab.get_entanglement_entropy(SYSTEM_1[1])
            entropyAC = tab.get_entanglement_entropy_for_separated_systems(SYSTEM_A, SYSTEM_C)
            entropyBC = tab.get_entanglement_entropy(SYSTEM_C[1], SYSTEM_B[0])

            mutual_infoAC = tab.get_bipartite_mutual_information(SYSTEM_A, SYSTEM_C)
            mutual_infoAD = tab.get_bipartite_mutual_information(SYSTEM_A, SYSTEM_D)
            mutual_infoBC = tab.get_bipartite_mutual_information(SYSTEM_B, SYSTEM_C)

            entanglement_negAC = tab.get_entanglement_negativity(system_A=SYSTEM_A, system_B=SYSTEM_C)
            entanglement_negAD = tab.get_entanglement_negativity(system_A=SYSTEM_A, system_B=SYSTEM_D)
            entanglement_negBC = tab.get_entanglement_negativity(system_A=SYSTEM_B, system_B=SYSTEM_C)

            # Inputting each observable value into respective array with time value
            eeB_data_set[time_step,0] = time_step + 1
            eeB_data_set[time_step, 1] = entropyB
            eeAC_data_set[time_step,0] = time_step + 1
            eeAC_data_set[time_step, 1] = entropyAC
            eeBC_data_set[time_step,0] = time_step + 1
            eeBC_data_set[time_step, 1] = entropyBC

            miAC_data_set[time_step,0] = time_step + 1
            miAC_data_set[time_step, 1] = mutual_infoAC
            miAD_data_set[time_step,0] = time_step + 1
            miAD_data_set[time_step, 1] = mutual_infoAD
            miBC_data_set[time_step,0] = time_step + 1
            miBC_data_set[time_step, 1] = mutual_infoBC

            enAC_data_set[time_step,0] = time_step + 1
            enAC_data_set[time_step, 1] = entanglement_negAC
            enAD_data_set[time_step,0] = time_step + 1
            enAD_data_set[time_step, 1] = entanglement_negAD
            enBC_data_set[time_step,0] = time_step + 1
            enBC_data_set[time_step, 1] = entanglement_negBC

        # Assigning time values to exported data sets
        if trial == 0:
            eeB_export_data[:,0] = eeB_data_set[:,0]
            eeAC_export_data[:,0] = eeAC_data_set[:,0]
            eeBC_export_data[:,0] = eeBC_data_set[:,0]

            miAC_export_data[:,0] = miAC_data_set[:,0]
            miAD_export_data[:,0] = miAD_data_set[:,0]
            miBC_export_data[:,0] = miBC_data_set[:,0]
            
            enAC_export_data[:,0] = enAC_data_set[:,0]
            enAD_export_data[:,0] = enAD_data_set[:,0]
            enBC_export_data[:,0] = enBC_data_set[:,0]

        # Creating dummy columns to concatenate observable values to export data arrays
        dummy_column_eeB = eeB_data_set[:,1] 
        dummy_column_eeAC = eeAC_data_set[:,1] 
        dummy_column_eeBC = eeBC_data_set[:,1] 

        dummy_column_miAC = miAC_data_set[:,1] 
        dummy_column_miAD = miAD_data_set[:,1]
        dummy_column_miBC = miBC_data_set[:,1]
        
        dummy_column_enAC = enAC_data_set[:,1] 
        dummy_column_enAD = enAD_data_set[:,1]
        dummy_column_enBC = enBC_data_set[:,1]   

        # Concatenating export arrays 
        eeB_export_data = np.concatenate([eeB_export_data, dummy_column_eeB[:,np.newaxis]], axis = 1)
        eeAC_export_data = np.concatenate([eeAC_export_data, dummy_column_eeAC[:,np.newaxis]], axis = 1)
        eeBC_export_data = np.concatenate([eeBC_export_data, dummy_column_eeBC[:,np.newaxis]], axis = 1)

        miAC_export_data = np.concatenate([miAC_export_data, dummy_column_miAC[:,np.newaxis]], axis = 1)
        miAD_export_data = np.concatenate([miAD_export_data, dummy_column_miAD[:,np.newaxis]], axis = 1)
        miBC_export_data = np.concatenate([miBC_export_data, dummy_column_miBC[:,np.newaxis]], axis = 1)

        enAC_export_data = np.concatenate([enAC_export_data, dummy_column_enAC[:,np.newaxis]], axis = 1)
        enAD_export_data = np.concatenate([enAD_export_data, dummy_column_enAD[:,np.newaxis]], axis = 1)
        enBC_export_data = np.concatenate([enBC_export_data, dummy_column_enBC[:,np.newaxis]], axis = 1)

        tic = time.perf_counter()
        print(f"That took {tic - toc} seconds for trial {trial}!")

    np.savez(f"{NUM_QUBITS}q_{SYSTEM_DIMENSION}dim_{RATE_ARRAY[rate]:.2f}rate_{NUM_TIME_STEPS}steps_{NUM_TRIALS}tr_{TODAY}", 
                eeB=eeB_export_data, eeAC=eeAC_export_data, eeBC=eeBC_export_data,
                miAC=miAC_export_data, miAD=miAD_export_data, miBC=miBC_export_data,
                enAC=enAC_export_data, enAD=enAD_export_data, enBC=enBC_export_data)

print("Test completed. :)")