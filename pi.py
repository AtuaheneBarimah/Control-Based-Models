import control as ct
import numpy as np
import pandas as pd

def Asset_PID(Kp,Ti,Td, plant, Sim, Pi_scale):
    num = np.array([Kp*Ti, Kp])
    den = np.array([Ti, 0])
    controller = ct.tf(num, den)
    time = np.linspace(0, 10, 10)
    Sim['P2_P1'] = ((abs(Sim.P2_bar-Sim.P1_bar)))*Pi_scale
    results = []
    for index, row in Sim.iterrows():
        reference = row['P2_P1']
        sys_cl = ct.feedback(ct.series(controller, plant), 1)
        t, y = ct.step_response(sys_cl, time)
        y_scaled = y * reference
        _, u = ct.step_response(ct.series(controller, 
                                        ct.feedback(1, ct.series(controller, plant))), 
                            time)
        u_scaled = u * reference
        u_required = reference
        
        # Store results
        results.append({
            'reference': reference,
            'y_scaled': y_scaled,
            'u_required': u_required,
            'u_actual': u_scaled,
            'tracking_error': 1 - y_scaled[-1] 
        })
    results_df = pd.DataFrame(results)

    return results_df.tracking_error
