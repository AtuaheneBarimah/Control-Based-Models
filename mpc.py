import control as ct
import numpy as np
import pandas as pd

def simulate_mpc_2(reference_df, K, T, simulation_steps=None):
    # Initialize MPC with system parameters
    mpc = MPCController_2(K=K, T=T)
    
    if simulation_steps is None:
        simulation_steps = len(reference_df)

    results = pd.DataFrame(
        index=range(simulation_steps),
        columns=['Time', 'y', 'u', 'reference']
    )
    
    # Initial conditions
    results.loc[0, 'y'] = 0.0
    if 'Time' in reference_df.columns:
        results.loc[0, 'Time'] = reference_df['Time'].iloc[0]
    else:
        results.loc[0, 'Time'] = 0.0
    
    for k in range(1, simulation_steps):
        # Calculate time step
        if k < len(reference_df):
            dt = reference_df['Time'].iloc[k] - reference_df['Time'].iloc[k-1]
        else:
            dt = reference_df['Time'].iloc[-1] - reference_df['Time'].iloc[-2]
        
        # Update time
        if 'Time' in reference_df.columns and k < len(reference_df):
            results.loc[k, 'Time'] = reference_df['Time'].iloc[k]
        else:
            results.loc[k, 'Time'] = results.loc[k-1, 'Time'] + dt
        
        # Reference sequence for horizon
        if k < len(reference_df):
            future_indices = range(k, min(k + mpc.horizon, len(reference_df)))
            y_ref_sequence = reference_df['P2_P1'].iloc[future_indices].values

            if len(y_ref_sequence) < mpc.horizon:
                y_ref_sequence = np.pad(
                    y_ref_sequence,
                    (0, mpc.horizon - len(y_ref_sequence)),
                    mode='edge'
                )
        else:
            y_ref_sequence = np.ones(mpc.horizon) * reference_df['P2_P1'].iloc[-1]
        
        # Get control input
        u = mpc.get_control_input(
            y_ref_sequence,
            results.loc[k-1, 'y'],
            dt
        )
        
        # Apply control and update system
        results.loc[k, 'u'] = u
        results.loc[k, 'y'] = system_model_2(
            results.loc[k-1, 'y'],
            u,
            mpc.K,
            mpc.T,
            dt
        )

        # Apply constraints
        results.loc[k, 'y'] = np.clip(results.loc[k, 'y'], y_min, y_max)
        results.loc[k, 'u'] = np.clip(results.loc[k, 'u'], mpc.u_min, mpc.u_max)

        # Store reference
        if k < len(reference_df):
            results.loc[k, 'reference'] = reference_df['P2_P1'].iloc[k]
        else:
            results.loc[k, 'reference'] = reference_df['P2_P1'].iloc[-1]
    
    return results
  
def filter_mpc_2(K, T, N, Sim, sampled_data, dis_size):
    y_min, y_max = 0.0, 1.0 
    
    result_dict = {}
    for idx, row in Sim.iterrows():
        t_test_1 = (abs(row['P2_bar'] / row['P1_bar']))+0.2
        t_test_2 = np.linspace(1, t_test_1, dis_size)
        Time_test_1 = row['Time']
        Time_test_2 = np.linspace(0, Time_test_1, dis_size)
        
        if len(Time_test_2) > 1:
            dt = Time_test_2[1] - Time_test_2[0]
        else:
            dt = 0.02 
            
        result_dict[idx] = {
            'Time': Time_test_2,
            'P2_P1': t_test_2,
            'dt': dt
        }
    
    df_with_arrays = pd.DataFrame.from_dict(result_dict, orient='index')
    df_with_arrays_each_row = {}

    for idx, row in df_with_arrays.iterrows():
        df_single = pd.DataFrame({
            'Time': row['Time'],
            'P2_P1': row['P2_P1']
        })
        df_with_arrays_each_row[idx] = df_single
        
    results_dict = {}
    for idx, df_single in df_with_arrays_each_row.items():
        dt = df_with_arrays.loc[idx, 'dt']
        results = simulate_mpc_2(df_single, K, T)
        results_dict[idx] = results

    all_results = []
    for idx, results in results_dict.items():
        results['Original_Index'] = idx
        all_results.append(results)
        final_results = pd.concat(all_results).reset_index(drop=True)

    MPC_output = round(float(final_results['y'].iloc[-1]), 2)
    #MPC_output = round(float(final_results['u'].max()), 2)
    MPC_output = (1*(MPC_output-0.0))/(1-0.0) 
    #MPC_output = 1-MPC_output  
    return MPC_output
