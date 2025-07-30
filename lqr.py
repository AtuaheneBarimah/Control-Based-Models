import control as ct
import numpy as np

def Asset_LQR(plant, Sim, Pi_scale):
    sys_ss = ct.tf2ss(plant)
    A, B, C, D = np.squeeze(sys_ss.A), np.squeeze(sys_ss.B), np.squeeze(sys_ss.C), np.squeeze(sys_ss.D)

    # LQR Design
    Q = np.array([[10.0]])  # State tracking penalty
    R = np.array([[1.0]])   # Control effort penalty
    K, S, E = ct.lqr(sys_ss, Q, R)
    print(f"LQR feedback gain K: {K.item():.4f}")

    # Time vector
    time = np.linspace(0, 10, 10)
    Sim['P2_P1'] = ((abs(Sim.P2_bar-Sim.P1_bar)))
    results = []
    for index, row in Sim.iterrows():
        reference = row['P2_P1']
        u_required = (-A/B) * reference 
        A_cl = A - B*K
        B_cl = B*K 
        C_cl = C
        D_cl = D
        t, y = ct.step_response(ct.ss(A_cl, B_cl, C_cl, D_cl), T=time)
        y_scaled = y * reference
        x = y_scaled  
        u = K.item() * (reference - x)  
        
        # Store results
        results.append({
            'reference': reference,
            'y_scaled': y_scaled,
            'u_required': u_required,
            'u_actual': u,
            'tracking_error': 1 - y_scaled[-1] 
        })
    results_df = pd.DataFrame(results)

    return results_df.tracking_error
