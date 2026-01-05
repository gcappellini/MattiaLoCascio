import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#======= CLASS DEFINITIONS =======#

class BranchNet(nn.Module):
    """Branch network: encodes function inputs from sensor measurements"""
    def __init__(self, n_sensors, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_sensors, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, sensors):
        return self.net(sensors)


class TrunkNet(nn.Module):
    """Trunk network: encodes spatiotemporal locations (x, t)"""
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # Input: [x, t]
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, xt):
        """
        Args:
            xt: (n_points, 2) with columns [x, t]
        """
        return self.net(xt)


class PINNDeepONet(nn.Module):
    """
    Physics-Informed DeepONet for heat equation with source:
    u_t = u_xx + f(x)
    
    Two-branch architecture:
    - Branch_IC: encodes initial condition u(x, 0)
    - Branch_Source: encodes heat source f(x)
    """
    
    def __init__(self, n_sensors_ic=20, n_sensors_src=20,n_sensors_thermal=20, 
                 branch_hidden=50, trunk_hidden=50, p=50, n_layers=2):
        super().__init__()



        self.n_layers = n_layers
        self.n_sensors_ic = n_sensors_ic
        self.n_sensors_src = n_sensors_src
        self.n_sensors_thermal = n_sensors_thermal
        self.p = p  
        self.domain = [0.0, 1.0]  # Spatial domain
        
        # Two branch networks
        self.branch_ic = BranchNet(n_sensors_ic, branch_hidden, p)
        self.branch_source = BranchNet(n_sensors_src, branch_hidden, p)
        self.branch_thermal_meas = BranchNet(n_sensors_thermal, branch_hidden, p)
        
        # Single trunk network
        self.trunk = TrunkNet(trunk_hidden, p)
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Fixed sensor locations
        #These functions create evenly spaced sensor locations in the domain [0.0, 1.0]
        self.register_buffer('sensor_x_ic', 
                            torch.linspace(self.domain[0], self.domain[1], n_sensors_ic))         #size (n_sensors_ic,)
        self.register_buffer('sensor_x_src', 
                           torch.linspace(self.domain[0], self.domain[1], n_sensors_src))         #size (n_sensors_src,)
        self.register_buffer('sensor_t_thermal_meas', 
                           torch.linspace(self.domain[0], self.domain[1], n_sensors_thermal))     #size (n_sensors_thermal,)
    
    def forward(self, ic_sensors, xt, src_sensors, thermal_sensors):
        """
        Args:
            ic_sensors: (batch, n_sensors_ic) or (n_sensors_ic,)
            src_sensors: (batch, n_sensors_src) or (n_sensors_src,)
            thermal_sensors: (batch, n_sensors_thermal) or (n_sensors_thermal,)
            xt: (n_points, 2) spatiotemporal coordinates [x, t]
        
        Returns:
            u: (batch, n_points) or (n_points,)
        """
        # Handle single sample
        single_sample_ic = ic_sensors.dim() == 1
        single_sample_src = src_sensors.dim() == 1
        single_sample_thermal = thermal_sensors.dim() == 1

        if single_sample_ic:
            ic_sensors = ic_sensors.unsqueeze(0)
        if single_sample_src:
            src_sensors = src_sensors.unsqueeze(0)
        if single_sample_thermal:
            thermal_sensors = thermal_sensors.unsqueeze(0)
        
        # Encode IC, source and measurements
        b_ic = self.branch_ic(ic_sensors)       # (batch, p)
        b_src = self.branch_source(src_sensors) # (batch, p)
        b_meas = self.branch_thermal_meas(thermal_sensors) # (batch, p)
        b_meas = b_meas.sum(dim=0, keepdim=True) # Sum over thermal sensors (batch, p)
        
        # Combine branch outputs
        b_combined = b_ic + b_src + b_meas              # (batch, p)
        
        # Encode spatiotemporal locations
        tau = self.trunk(xt)                    # (n_points, p)
        
        # DeepONet operation: (batch, p) @ (p, n_points) = (batch, n_points)
        u_net = torch.matmul(b_combined, tau.T) + self.bias  # (batch, n_points)
        
        # Extract x,t coordinates
        x = xt[:, 0]  # (n_points,)
        t = xt[:, 1]  # (n_points,)

        
        # Hard BC - IC enforcement: u = (x - 1) * t * u_net
        # At  x=1: u=0
        bc_factor = (x - 1.0) * t  # (n_points,)
        
        # Apply BC factor (broadcast over batch dimension)
        if single_sample_ic or single_sample_src or single_sample_thermal:
            u = bc_factor * u_net  
        else:
            u = bc_factor.unsqueeze(0) * u_net # (batch, n_points)
        
        return u.squeeze(0) if single_sample_ic and single_sample_src and single_sample_thermal else u
    
    def compute_coefficients(self, x, L0 = 0.07, t_span = 1800.035, c = [2348,3421], ro = [911,1090], k = [0.21,0.49],
                             b4 = 0.829, y2_0 = 30.2, Tmin = 21.5, x0 = 0.004, PD = 0.0136, beta = 1,deltaT = None, v = None):
        """
        Compute coefficients a1, a2, a3, a4 based on position x
        x: (n_points,)
        Returns:
            a1: (n_points,)
            a2: (n_points,)
            a3: (n_points,)
            a4: (n_points,)
        """
        # indice di strato per ogni x
        i = torch.clamp((x * self.n_layers).long(),
                        0, self.n_layers - 1)
        
        # Convert parameters to tensors
        L0 = torch.as_tensor(L0, dtype=torch.float32, device=x.device)
        t_span = torch.as_tensor(t_span, dtype=torch.float32, device=x.device)
        x0 = torch.as_tensor(x0, dtype=torch.float32, device=x.device)
        beta = torch.as_tensor(beta, dtype=torch.float32, device=x.device)
        
        deltaT = (y2_0 - Tmin)/b4
        v = np.log(2/(PD-10**(-2)* x0))

        v  = torch.as_tensor(v,  dtype=torch.float32, device=x.device)
        deltaT = torch.as_tensor(deltaT, dtype=torch.float32, device=x.device)
        # liste python → tensori torch
        ro_vec = torch.as_tensor(ro, dtype=torch.float32, device=x.device)
        k_vec  = torch.as_tensor(k,  dtype=torch.float32, device=x.device)
        c_vec = torch.as_tensor(c, dtype=torch.float32, device=x.device)

        # coefficiente corretto per OGNI x
        ro_i = ro_vec[i]    # shape (Ns,)
        k_i  = k_vec[i]     # shape (Ns,)
        c_i  = c_vec[i]     # shape (Ns,)

        a1 = (L0**2 * ro_i * c_i) / (k_i * t_span)
        a2 = L0**2 * ro_i * c_i / k_i
        a3 = ((ro_i*L0**2)/(k_i*deltaT))*beta*np.exp(v*x0)
        a4 = v * L0

        # Array to Tensor
        a1 = torch.as_tensor(a1, dtype=torch.float32, device=x.device)
        a2 = torch.as_tensor(a2, dtype=torch.float32, device=x.device)
        a3 = torch.as_tensor(a3, dtype=torch.float32, device=x.device)
        a4 = torch.as_tensor(a4, dtype=torch.float32, device=x.device)

        return a1, a2, a3, a4

    def compute_pde_residual(self, ic_sensors, xt, src_values, thermal_sensors, src_sensors, a1=18.992, a2=34185.667, wb=0.0005):
        """
        Compute PDE residual: R = u_t - u_xx - f(x)
        
        Args:
            ic_sensors: (n_sensors_ic,) IC measurements
            src_sensors: (n_sensors_src,) source measurements
            xt: (n_points, 2) collocation points [x, t]
            src_values: (n_points,) source values f(x) at collocation points
        
        Returns:
            residual: (n_points,)
        """
        #Parameters
        x = xt[:, 0]

        # Compute coefficients at collocation points
        a1 , a2, a3, a4 = self.compute_coefficients(x)

        # Enable gradients for xt
        xt_requires_grad = xt.clone().requires_grad_(True) 
            #.clone() generates a copy of the tensor xt that requires gradient computation.
            # .requires_grad_(True) sets the requires_grad attribute of the tensor to True,
            # indicating that we want to compute gradients with respect to this tensor during the automatic differentiation.
        
        # Forward pass
        u = self.forward(ic_sensors, xt_requires_grad, src_sensors, thermal_sensors)
        
        # Compute u_t using autograd
        u_t = torch.autograd.grad(u, xt_requires_grad, 
                                  torch.ones_like(u),
                                  create_graph=True)[0][:, 1]  # ∂u/∂t
        
        # Compute u_x
        u_x = torch.autograd.grad(u, xt_requires_grad,
                                  torch.ones_like(u),
                                  create_graph=True)[0][:, 0]  # ∂u/∂x
        
        # Compute u_xx
        u_xx = torch.autograd.grad(u_x, xt_requires_grad,
                                   torch.ones_like(u_x),
                                   create_graph=True)[0][:, 0]  # ∂²u/∂x²
        
        # PDE residual: u_t - u_xx - f(x)
        residual = a1 * u_t - u_xx + wb * a2 * u - src_values
        
        return residual
 

    def compute_bc0_residual(self, ic_sensors, src_sensors, thermal_sensors, xt, a5=175):
        """
        Compute bc0 residual: R = u_t - u_xx - f(x)
        
        Args:
            ic_sensors: (n_sensors_ic,) IC measurements
            src_sensors: (n_sensors_src,) source measurements
            xt: (n_points, 2) collocation points [x, t]
        
        Returns:
            residual: (n_points,)
        """
        xt_requires_grad = xt.clone().requires_grad_(True)
        
        # Forward pass
        u = self.forward(ic_sensors, xt_requires_grad, src_sensors, thermal_sensors)
        
        # Compute u_x
        u_x = torch.autograd.grad(u, xt_requires_grad,
                                  torch.ones_like(u),
                                  create_graph=True)[0][:, 0]  # ∂u/∂x
        
        # BC0 residual: u_x - f(x)
        t = xt[:, 1]
        y3 = 0.65*(1-torch.exp(-t/0.5))
        residual = u_x - a5*(y3 - u)
        
        return residual
    
    def compute_thermal_meas_residual(self,ic_sensors, src_sensors, thermal_sensors,x_meas,xt,Tx):
        """
        Compute bc0 residual: R = u(x_meas) -  1 - torch.exp(-t / Tx)
        
        Args:
            thermal_meas_sensors: (thermal_meas_sensors,) internal measurements
            xt: (n_points, 2) collocation points [x, t]
        
        Returns:
            residual: (n_points,)
        """
        t = xt[:, 1]
        x = xt[:, 0]
        #Internal Measurement
        u_meas = self.generate_internal_measurements(Tx, t)    #(batch, n_therm)
        

        # Forward pass
        u = self.forward(ic_sensors, xt, src_sensors,thermal_sensors) #

        # Thermal measurement residual: u(x_meas) - analytical solution
        #u_meas_pred = u[x == x_meas]
        u_meas_pred = u
        residual = u_meas - u_meas_pred
        return residual       
    
    def generate_ic(self, theta_gt, theta0_0, X_gt=0.14286, a5=175, y3_0=0.0, x = None):
        """
        Generate initial condition sensors based on cubic polynomial coefficients.
        Solves for b1, b2, b3 in the system:
            b1 + b2 + b3 + y2_0 = 0
            b1*X_gt**3 + b2*X_gt**2 + b3*X_gt + y2_0 = theta_gt
            b3 = a5*(y3_0 - y2_0)
        Returns:
            ic_sensors: tensor of initial condition values at sensor locations
        """

        # Evaluate cubic at sensor locations
        if x is None:
            x = self.sensor_x_ic
        ic = 0 * torch.ones_like(x)
        return ic
    
    
    def generate_source(self,source_type='gaussian', amplitude=0.5, x = None, a3 = 0.7939, a4 = 3.570):
        """
        Generate heat source f(x)
        
        Args:
            source_type: 'gaussian', 'sine', 'constant'
            amplitude: source strength
        
        Returns:
            src_sensors: source values at sensor locations
        """
        if x is None:
            x = self.sensor_x_src

        # Compute coefficients at sensor locations
        a1 , a2, a3, a4 = self.compute_coefficients(x)

        # Evaluate source at sensor locations
        if source_type == 'SAR':
            # SAR source profile
            src = amplitude *a3* torch.exp(-a4*x)
        elif source_type == 'gaussian':
            # Gaussian centered at x=0.50
            center = 0.50
            width = 0.3
            src = amplitude * torch.exp(-((x - center) / width) ** 2)
        elif source_type == 'sine':
            # Sine wave
            src = amplitude * torch.sin(2 * np.pi * x)
        elif source_type == 'constant':
            # Constant source
            src = amplitude * torch.ones_like(x)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        return src
    
    def generate_internal_measurements(self, Tx, t=None):
        """
        Generate internal temperature measurements at fixed location x_meas
        using the analytical solution of the heat equation with source.
        
        Args:
            Tx: parameter that enables to fit temperature distribution over time at x=x_meas
            t: time tensor
            u: analytical solution at x_meas over time t
        
        Returns:
            thermal_meas_sensors: temperature values at measurement sensors
        """
        if t is None:
            t = self.sensor_t_thermal_meas
        # Analytical solution at x_meas over time t
        Tx = torch.as_tensor(Tx, dtype=torch.float32, device=t.device)
        if Tx.numel() == 1:   #single sample
            u = 1 - torch.exp(-t / Tx)     #(len(t),)
        else:    #batch
            Tx.unsqueeze_(0)   #(1, batch)
            u = torch.zeros((Tx.shape[1], t.numel()), dtype=torch.float32, device=t.device)        #(batch, len(t))
            for i in range(Tx.shape[1]):
                u[i, :] = 1 - torch.exp(-t / Tx[0,i])
        return u
    
    def train_pinn(self, n_epochs=5000, n_colloc=200, n_bc0 = 40, n_ic = 50, n_therm = 50,lr=1e-3,
                    theta_gt_range=(0.7, 1.0), theta0_0_range=(0.6, 0.9), source_type='gaussian', source_amplitude=0.0,
                    Tx_range1=(0.0,0.5), Tx_range2 = (0.5,1.0),Tx_range3 = (1.0,1.5), x_meas=[0.2, 0.5, 0.8]):
        """
        Train PINN-DeepONet with physics-informed loss
        
        Loss = w_pde * L_pde + w_bc0 * L_bc0 + w_meas * L_meas
        Note: BC1 and IC are enforced as hard constraint, no soft BC and IC losses needed
        """
        print("="*70)
        print("TRAINING PHYSICS-INFORMED DEEPONET")
        print("="*70)
        print(f"Epochs: {n_epochs}")
        print(f"Collocation points: {n_colloc}")
        print(f"Source type: {source_type}, Amplitude: {source_amplitude}")
        print("Boundary conditions: HARD constraints (no soft loss)")
        print("="*70)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=500
        )

        
        history = {
            'total': [], 'pde': [], 'bc0': [], 'meas': []
        }
        
        for epoch in range(n_epochs):
            # Sample random IC amplitude
            theta_gt = theta_gt_range[0] + (theta_gt_range[1] - theta_gt_range[0]) * torch.rand(1).item()
            theta0_0 = theta0_0_range[0] + (theta0_0_range[1] - theta0_0_range[0]) * torch.rand(1).item()
            Tx1 = Tx_range1[0] + (Tx_range1[1] - Tx_range1[0]) * torch.rand(1).item()
            Tx2 = Tx_range2[0] + (Tx_range2[1] - Tx_range2[0]) * torch.rand(1).item()
            Tx3 = Tx_range3[0] + (Tx_range3[1] - Tx_range3[0]) * torch.rand(1).item()
            Tx_array = torch.tensor([Tx1,Tx2,Tx3], dtype=torch.float32)         # (3,)

            # Generate IC and source sensors
            ic_sensors = self.generate_ic(theta_gt, theta0_0)                     #[n_sensors_ic,]
            src_sensors = self.generate_source(source_type, source_amplitude)     #[n_sensors_src,]
            thermal_sensors = self.generate_internal_measurements(Tx_array)             #[batch, n_sensors_thermal]
            
            # Sample collocation points (x, t)
            x_colloc = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc)
            t_colloc = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc)
            xt_colloc = torch.stack([x_colloc, t_colloc], dim=1)                 #[n_colloc, 2]
            
            # Source values at collocation points
            src_colloc = self.generate_source(source_type, source_amplitude, x_colloc)        #[n_colloc,]
            
            # === PDE Loss ===
            residual = self.compute_pde_residual(ic_sensors,  xt_colloc, src_colloc, thermal_sensors,src_sensors)
            loss_pde = torch.mean(residual ** 2)

            # === BC0 Loss ===
            t_bc0 = torch.linspace(self.domain[0], self.domain[1], n_bc0)
            x_bc0 = torch.zeros(n_bc0)
            xt_bc0 = torch.stack([x_bc0, t_bc0], dim=1)            #[n_bc0, 2]
            residual_bc0 = self.compute_bc0_residual(ic_sensors, src_sensors, thermal_sensors, xt_bc0)
            loss_bc0 = torch.mean(residual_bc0 ** 2)
            
            # === Thermal Measurement Loss ===
            t_meas = torch.linspace(self.domain[0], self.domain[1], n_therm)         #[n_therm,]

            # Measurements at x_meas = 0.2
            x_meas1 = x_meas[0] * torch.ones(n_therm)
            xt_therm1 = torch.stack([x_meas1, t_meas], dim=1)          #[n_therm, 2]
            residual_meas1 = self.compute_thermal_meas_residual(ic_sensors, src_sensors, thermal_sensors,x_meas1,xt_therm1,Tx_array[0,0])
            loss_meas1 = torch.mean(residual_meas1 ** 2)

            # Measurements at x_meas = 0.5
            x_meas2 = x_meas[1] * torch.ones(n_therm)
            xt_therm2 = torch.stack([x_meas2, t_meas], dim=1)          #[n_therm, 2]
            residual_meas2 = self.compute_thermal_meas_residual(ic_sensors, src_sensors, thermal_sensors,x_meas2,xt_therm2,Tx_array[0,1])
            loss_meas2 = torch.mean(residual_meas2 ** 2)

            # Measurements at x_meas = 0.8
            x_meas3 = x_meas[2] * torch.ones(n_therm)
            xt_therm3 = torch.stack([x_meas3, t_meas], dim=1)          #[n_therm, 2]
            residual_meas3 = self.compute_thermal_meas_residual(ic_sensors, src_sensors, thermal_sensors,x_meas3,xt_therm3,Tx_array[0,2])
            loss_meas3 = torch.mean(residual_meas3 ** 2)

            loss_meas = loss_meas1 + loss_meas2 + loss_meas3

                    
            # Loss weights (no BC weight needed - hard constraint!)

            if epoch == 0:
                initial_losses = torch.tensor([loss_pde.item(), loss_bc0.item(), loss_meas.item()])
            w_pde = 5.0 / (initial_losses[0])
            w_bc0 = 5.0 / (initial_losses[1])
            w_meas = 5.0 / (initial_losses[2])

            # === Total Loss (no BC1 loss - hard constraint!) ===
            loss_total = w_pde * loss_pde + w_bc0 * loss_bc0 + w_meas * loss_meas
            
            # Optimization step
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Record history
            history['total'].append(loss_total.item())
            history['pde'].append(loss_pde.item())
            history['bc0'].append(loss_bc0.item())
            history['meas'].append(loss_meas.item())
            
            scheduler.step(loss_total)
            
            # Print progress
            if epoch % 500 == 0 or epoch == n_epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:5d} | Loss: {loss_total.item():.6f} | "
                      f"PDE: {loss_pde.item():.6f} | BC0: {loss_bc0.item():.6f} | "
                      f"Meas: {loss_meas.item():.6f} | LR: {current_lr:.2e}")
        
        print("\n✓ Training complete!\n")
        return history
    
# ------- Plotting ------- #
def plot_solution(model, theta_gt_test, theta0_0_test, Tx_test, x_meas_test, source_type='gaussian', source_amplitude=0.5, gt=None):
    """Visualize the trained solution"""

    Tx_test = torch.tensor(Tx_test, dtype=torch.float32)    #(3,)
    
    # Generate test case
    ic_sensors = model.generate_ic(theta_gt_test, theta0_0_test)
    src_sensors = model.generate_source(source_type, source_amplitude)
    thermal_sensors = model.generate_internal_measurements(Tx_test)
    
    # Create spatiotemporal grid
    nx, nt = 101, 101
    x_plot = torch.linspace(model.domain[0], model.domain[1], nx)
    t_plot = torch.linspace(model.domain[0], model.domain[1], nt)
    
    X, T = torch.meshgrid(x_plot, t_plot, indexing='ij')
    xt_grid = torch.stack([X.flatten(), T.flatten()], dim=1)
    
    # Predict solution
    with torch.no_grad():
        u_pred = model.forward(ic_sensors, xt_grid, src_sensors, thermal_sensors)
        u_pred = u_pred.reshape(nx, nt).numpy()

    # Comparison with ground truth
    if gt is not None:
        u_gt = gt[:, 2].reshape(nx,nt, order = 'F') 
        abs_error = np.abs(u_pred - u_gt)   #Absolute errror
    
    # Source function for plotting
    src_plot = model.generate_source(source_type, source_amplitude, x_plot).numpy()
    
    # Plotting
    fig = plt.figure(figsize=(18, 5))
    
    # Plot 1: Solution evolution (snapshots)
    ax1 = plt.subplot(1, 3, 1)
    time_indices = [0, nt//4, nt//2, 3*nt//4, -1]
    for idx in time_indices:
        t_val = t_plot[idx].item()
        color = plt.cm.tab10(time_indices.index(idx))
        ax1.plot(x_plot.numpy(), u_pred[:, idx], label=f't={t_val:.3f}', color=color)
        ax1.plot(x_plot.numpy(), u_gt[:, idx], '--', linewidth=1, color=color)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('u(x,t)', fontsize=12)
    ax1.set_title(f'Solution Evolution (theta_gt,0_0={theta_gt_test,theta0_0_test})', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spatiotemporal heatmap
    ax2 = plt.subplot(1, 3, 2)
    im = ax2.contourf(T.numpy(), X.numpy(), u_pred, levels=20, cmap='RdBu_r')
    ax2.set_xlabel('t', fontsize=12)
    ax2.set_ylabel('x', fontsize=12)
    ax2.set_title('Solution u(x,t)', fontsize=12)
    plt.colorbar(im, ax=ax2)
    
    # Plot 3: Source function
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(x_plot.numpy(), src_plot, 'r-', linewidth=2, label='f(x)')
    ic_plot = model.generate_ic(theta_gt_test,theta0_0_test, x = x_plot).numpy()
    ax3.plot(x_plot.numpy(), ic_plot, 'b--', linewidth=2, label='IC: u(x,0)')
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Initial Condition', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Ground truth absolute error (if ground truth provided)
    if gt is not None:
        import matplotlib.colors as mcolors

        fig_err = plt.figure(figsize=(6, 5))

        #======== Absolute error heatmap ========#
        ax_err = fig_err.add_subplot(1, 1, 1)
        # White -> Red colormap so minimum is white and maximum is red
        cmap = mcolors.LinearSegmentedColormap.from_list('white_red', ['white', 'red'])
        # Ensure red corresponds to the maximum absolute error
        vmin = 0.0
        vmax = float(abs_error.max())

        im_err = ax_err.contourf(T.numpy(), X.numpy(), abs_error, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_err.set_xlabel('t', fontsize=12)
        ax_err.set_ylabel('x', fontsize=12)
        ax_err.set_title('Absolute error |u_pred - u_gt|', fontsize=12)
        cbar = plt.colorbar(im_err, ax=ax_err)
        cbar.set_label('Absolute error')

    # Plot 5: u(x_meas,t) predicted vs measured one for multiple Tx_test
    fig_4, axes = plt.subplots(1, 3, figsize=(15, 6))

    for i in range(len(x_meas_test)):
        axes[i].plot(t_plot.numpy(), u_pred[int(round(x_meas_test[i] * 100)), :].squeeze(), label=f'Predicted u(x_meas,t)')
        u_meas_true = 1 - np.exp(-t_plot.numpy() / Tx_test[0,i])
        axes[i].plot(t_plot.numpy(), u_meas_true, 'r--', label='Real u(x_meas,t)')
        axes[i].set_xlabel('t', fontsize=12)
        axes[i].set_ylabel(f'u({x_meas_test[i]},t)', fontsize=12)
        axes[i].set_title(f'Internal Measurement at x={x_meas_test[i]} for Tx={Tx_test[0,i]:.2f}', fontsize=12)
        axes[i].legend()
    
    
    plt.tight_layout()
    return fig, fig_err, fig_4



def plot_training_history(history):
    """Plot training loss history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0,0].semilogy(history['total'], 'k-', linewidth=1.5)
    axes[0,0].set_title('Total Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].semilogy(history['pde'], 'b-', linewidth=1.5)
    axes[0,1].set_title('PDE Residual Loss')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].semilogy(history['meas'], 'm-', linewidth=1.5)
    axes[1,0].set_title('Thermal Measurement Loss')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].grid(True, alpha=0.3)

    axes[1,1].semilogy(history['bc0'], 'g-', linewidth=1.5)
    axes[1,1].set_title('BC0 Loss')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig




#======= MAIN HYDRA =======#
import hydra
from omegaconf import DictConfig
import os
from hydra.utils import to_absolute_path

@hydra.main(config_path="conf", config_name="config", version_base=None)

def main(cfg: DictConfig):
    
    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Create model
    model = PINNDeepONet(
        n_sensors_ic=cfg.model.n_sensors_ic,
        n_sensors_src=cfg.model.n_sensors_src,
        n_sensors_thermal=cfg.model.n_sensors_thermal,
        branch_hidden=cfg.model.branch_hidden,
        trunk_hidden=cfg.model.trunk_hidden,
        p=cfg.model.p,
        n_layers=cfg.model.n_layers
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Train
    history = model.train_pinn(
        n_epochs=cfg.training.n_epochs,
        n_colloc=cfg.training.n_colloc,
        n_bc0=cfg.training.n_bc0,
        n_ic=cfg.training.n_ic,
        n_therm=cfg.training.n_therm,
        lr=cfg.training.lr,
        theta_gt_range=cfg.training.theta_gt_range,
        theta0_0_range=cfg.training.theta0_0_range,
        source_type=cfg.source.source_type,
        source_amplitude=cfg.source.source_amplitude_multilayer,
        Tx_range1=cfg.training.Tx_range1_multilayer,
        Tx_range2=cfg.training.Tx_range2_multilayer,
        Tx_range3=cfg.training.Tx_range3_multilayer,
        x_meas=cfg.training.x_multiple_meas
    )
    
    # Load ground truth data for comparison
    gt_data = np.loadtxt(to_absolute_path('gt_bioheat1D_src_multilayer.csv'))

    #Save model
    cur_dir = os.getcwd()
    print("Current working directory:", cur_dir)

    model_path = os.path.join(cur_dir, "pinn_deeponet_heat_with_src_3branch_multilayer_3internal_meas.pth")
    torch.save(model.state_dict(), model_path)
    print("✔ Model saved:", "pinn_deeponet_heat_with_src_multilayer_3internal_meas.pth")

    # Plot results
    print("Generating plots...")
    
    fig1 = plot_training_history(history)
    fig1_path = os.path.join(cur_dir, "pinn_deeponet_training_with_src_3branch_multilayer_3internal_meas.png")
    fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
    print("✔ Training plot saved:", "pinn_deeponet_training_with_src_multilayer_3internal_meas.png")


    fig3,fig_err,fig_4 = plot_solution(model, theta_gt_test = cfg.test.theta_gt_test, theta0_0_test = cfg.test.theta0_0_test, 
                                 Tx_test = cfg.test.Tx_test_multilayer_multiple_meas, x_meas_test = cfg.test.x_multiple_meas_test,source_type=cfg.source.source_type,
                                   source_amplitude = cfg.source.source_amplitude_multilayer, gt=gt_data)
    fig3_path = os.path.join(cur_dir, "pinn_deeponet_solution_with_src_3branch_multilayer_3internal_meas.png")
    fig_err_path = os.path.join(cur_dir, "comparison_src_3branch_multilayer_3internal_meas.png")
    fig_4_path = os.path.join(cur_dir, "internal_measurement_multilayer_3internal_meas.png")
    fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
    fig_err.savefig(fig_err_path, dpi=150, bbox_inches='tight')
    fig_4.savefig(fig_4_path, dpi=150, bbox_inches='tight')
    print("✔ Solution plot saved:", "pinn_deeponet_solution_with_src_multilayer_3internal_meas.png")
    print("✔ Comparison plot saved:", "comparison_src_3branch_multilayer_3internal_meas.png")
    print("✔ Internal measurement plot saved:", "internal_measurement_multilayer_3internal_meas.png")

    print("Files saved in:", cur_dir)
    plt.show()

if __name__ == "__main__":
    main()

# python pinn_deeponet_heat_3branch_with_src_multilayer.py -m     for multirun with different n_ic_sensors

# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\venv\Scripts\Activate.ps1