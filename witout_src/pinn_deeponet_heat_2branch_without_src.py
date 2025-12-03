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
    
    def __init__(self, n_sensors_ic=20, n_sensors_src=20, 
                 branch_hidden=50, trunk_hidden=50, p=50):
        super().__init__()
        
        self.n_sensors_ic = n_sensors_ic
        self.n_sensors_src = n_sensors_src
        self.p = p
        self.domain = [0.0, 1.0]  # Spatial domain
        
        # Two branch networks
        self.branch_ic = BranchNet(n_sensors_ic, branch_hidden, p)
        
        # Single trunk network
        self.trunk = TrunkNet(trunk_hidden, p)
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Fixed sensor locations
        #These functions create evenly spaced sensor locations in the domain [0.0, 1.0]
        self.register_buffer('sensor_x_ic', 
                            torch.linspace(self.domain[0], self.domain[1], n_sensors_ic))

    def forward(self, ic_sensors, xt, src_sensors = None):
        """
        Args:
            ic_sensors: (batch, n_sensors_ic) or (n_sensors_ic,)
            src_sensors: (batch, n_sensors_src) or (n_sensors_src,)
            xt: (n_points, 2) spatiotemporal coordinates [x, t]
        
        Returns:
            u: (batch, n_points) or (n_points,)
        """
        # Handle single sample
        single_sample = ic_sensors.dim() == 1
        if single_sample:
            ic_sensors = ic_sensors.unsqueeze(0)
            #src_sensors = src_sensors.unsqueeze(0)
        
        # Encode IC and source
        b_ic = self.branch_ic(ic_sensors)       # (batch, p)
        
        # Combine branch outputs
        b_combined = b_ic # + b_src               # (batch, p)
        
        # Encode spatiotemporal locations
        tau = self.trunk(xt)                    # (n_points, p)
        
        # DeepONet operation: (batch, p) @ (p, n_points) = (batch, n_points)
        u_net = torch.matmul(b_combined, tau.T) + self.bias
        
        # Extract x coordinates
        x = xt[:, 0]  # (n_points,)
        
        # Hard BC enforcement: u = (x - 1) * u_net
        # At  x=1: u=0
        bc_factor = (x - 1.0)  # (n_points,)
        
        # Apply BC factor (broadcast over batch dimension)
        if single_sample:
            u = bc_factor * u_net
        else:
            u = bc_factor.unsqueeze(0) * u_net  # (batch, n_points)
        
        return u.squeeze(0) if single_sample else u
    
    def compute_pde_residual(self, ic_sensors, xt, src_values = None, src_sensors = None, a1=18.992, a2=34185.667, wb=0.0005):
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
        xt_requires_grad = xt.clone().requires_grad_(True) 
            #.clone() generates a copy of the tensor xt that requires gradient computation.
            # .requires_grad_(True) sets the requires_grad attribute of the tensor to True,
            # indicating that we want to compute gradients with respect to this tensor during the automatic differentiation.
        
        # Forward pass
        u = self.forward(ic_sensors, xt_requires_grad)
        
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
        residual = a1 * u_t - u_xx + wb * a2 * u #- src_values
        
        return residual
    
    def compute_bc0_residual(self, ic_sensors, xt, a5=1.1666667, y3_0=0.0, src_sensors = None):
        """
        Compute bc0 residual: R = u_t - u_xx - f(x)
        
        Args:
            ic_sensors: (n_sensors_ic,) IC measurements
            src_sensors: (n_sensors_src,) source measurements
            xt: (n_points, 2) collocation points [x, t]
            src_values: (n_points,) source values f(x) at collocation points
        
        Returns:
            residual: (n_points,)
        """
        xt_requires_grad = xt.clone().requires_grad_(True)
        
        # Forward pass
        u = self.forward(ic_sensors, xt_requires_grad, src_sensors)
        
        # Compute u_x
        u_x = torch.autograd.grad(u, xt_requires_grad,
                                  torch.ones_like(u),
                                  create_graph=True)[0][:, 0]  # ∂u/∂x
        
        # BC0 residual: u_x - f(x)
        residual = u_x - a5*(y3_0 - u)
        
        return residual
    
    def generate_ic(self, theta_gt, theta0_0, X_gt=0.14286, a5=1.1666667, y3_0=0.0, x = None):
        """
        Generate initial condition sensors based on cubic polynomial coefficients.
        Solves for b1, b2, b3 in the system:
            b1 + b2 + b3 + y2_0 = 0
            b1*X_gt**3 + b2*X_gt**2 + b3*X_gt + y2_0 = theta_gt
            b3 = a5*(y3_0 - y2_0)
        Returns:
            ic_sensors: tensor of initial condition values at sensor locations
        """
        # Solve for b3
        b3 = - a5 * (y3_0 - theta0_0)
        # # System:
        # # eq1: b1 + b2 + b3 + theta0_0 = 0
        # # eq2: b1*X_gt**3 + b2*X_gt**2 + b3*X_gt + theta0_0 = theta_gt

        # # Substitute b3 into eq1 and eq2
        # # eq1: b1 + b2 = -b3 - theta0_0
        # # eq2: b1*X_gt**3 + b2*X_gt**2 = theta_gt - theta0_0 - b3*X_gt

        # Solve linear system for b1, b2
        A = np.array([
            [1, 1],
            [X_gt**3, X_gt**2]
        ])
        rhs = np.array([
            -b3 - theta0_0,
            theta_gt - theta0_0 - b3*X_gt
        ])
        b1, b2 = np.linalg.solve(A, rhs)

        # b4 = theta0_0 (constant term)
        b4 = theta0_0

        # Evaluate cubic at sensor locations
        if x is None:
            x = self.sensor_x_ic
        ic = b1 * x**3 + b2 * x**2 + b3 * x + b4
        return ic

    
    def train_pinn(self, n_epochs=5000, n_colloc=200, n_bc0 = 40, n_ic = 50, lr=1e-3, 
                   theta_gt_range=(0.7, 1.0), theta0_0_range=(0.6, 0.9)):
        """
        Train PINN-DeepONet with physics-informed loss
        
        Loss = w_pde * L_pde + w_ic * L_ic
        Note: BC is enforced as hard constraint, no soft BC loss needed
        """
        print("="*70)
        print("TRAINING PHYSICS-INFORMED DEEPONET")
        print("="*70)
        print(f"Epochs: {n_epochs}")
        print(f"Collocation points: {n_colloc}")
        print(f"IC amplitude range: {theta_gt_range}")
        print(f"Source type: None, amplitude: 0.0")
        print("Boundary conditions: HARD constraints (no soft loss)")
        print("="*70)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=500
        )
        
        # Loss weights (no BC weight needed - hard constraint!)
        w_pde = 1.0
        w_bc0 = 1.0
        w_ic = 10.0
        
        history = {
            'total': [], 'pde': [], 'ic': [], 'bc0': []
        }
        
        for epoch in range(n_epochs):
            # Sample random IC amplitude
            theta_gt = theta_gt_range[0] + (theta_gt_range[1] - theta_gt_range[0]) * torch.rand(1).item()
            theta0_0 = theta0_0_range[0] + (theta0_0_range[1] - theta0_0_range[0]) * torch.rand(1).item()

            # Generate IC and source sensors
            ic_sensors = self.generate_ic(theta_gt, theta0_0)
            
            # Sample collocation points (x, t)
            x_colloc = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc)
            t_colloc = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc)
            xt_colloc = torch.stack([x_colloc, t_colloc], dim=1)
         
            # === PDE Loss ===
            residual = self.compute_pde_residual(ic_sensors,  xt_colloc)
            loss_pde = torch.mean(residual ** 2)

            # === BC0 Loss ===
            t_bc0 = torch.linspace(self.domain[0], self.domain[1], n_bc0)
            x_bc0 = torch.zeros(n_bc0)
            xt_bc0 = torch.stack([x_bc0, t_bc0], dim=1)
            residual_bc0 = self.compute_bc0_residual(ic_sensors,  xt_bc0)
            loss_bc0 = torch.mean(residual_bc0 ** 2)
            
            # === IC Loss ===
            x_ic = torch.linspace(self.domain[0], self.domain[1], n_ic)
            t_ic = torch.zeros(n_ic)
            xt_ic = torch.stack([x_ic, t_ic], dim=1)
            
            u_ic_pred = self.forward(ic_sensors, xt_ic)
            u_ic_true = self.generate_ic(theta_gt, theta0_0,x = x_ic )
            loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)
            
            # === Total Loss (no BC1 loss - hard constraint!) ===
            loss_total = w_pde * loss_pde + w_ic * loss_ic + w_bc0 * loss_bc0
            
            # Optimization step
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Record history
            history['total'].append(loss_total.item())
            history['pde'].append(loss_pde.item())
            history['ic'].append(loss_ic.item())
            history['bc0'].append(loss_bc0.item())
            
            scheduler.step(loss_total)
            
            # Print progress
            if epoch % 500 == 0 or epoch == n_epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:5d} | Loss: {loss_total.item():.6f} | "
                      f"PDE: {loss_pde.item():.6f} | BC0: {loss_bc0.item():.6f} | IC: {loss_ic.item():.6f} | "
                      f"LR: {current_lr:.2e}")
        
        print("\n✓ Training complete!\n")
        return history
    
# ------- Plotting ------- #
def plot_solution(model, theta_gt_test, theta0_0_test, gt=None):
    """Visualize the trained solution"""
    
    # Generate test case
    ic_sensors = model.generate_ic(theta_gt_test, theta0_0_test)
    
    # Create spatiotemporal grid
    nx, nt = 101, 101
    x_plot = torch.linspace(model.domain[0], model.domain[1], nx)
    t_plot = torch.linspace(model.domain[0], model.domain[1], nt)
    
    X, T = torch.meshgrid(x_plot, t_plot, indexing='ij')
    xt_grid = torch.stack([X.flatten(), T.flatten()], dim=1)
    
    # Predict solution
    with torch.no_grad():
        u_pred = model.forward(ic_sensors, xt_grid)
        u_pred = u_pred.reshape(nx, nt).numpy()

    # Comparison with ground truth
    if gt is not None:
        u_gt = gt[:, 2].reshape(nx,nt, order = 'F') 
        abs_error = np.abs(u_pred - u_gt)   #Absolute errror
      
    # Plotting
    fig = plt.figure(figsize=(18, 5))
    
    # Plot 1: Solution evolution (snapshots)
    ax1 = plt.subplot(1, 3, 1)
    time_indices = [0, nt//4, nt//2, 3*nt//4, -1]
    for idx in time_indices:
        t_val = t_plot[idx].item()
        ax1.plot(x_plot.numpy(), u_pred[:, idx], label=f't={t_val:.3f}')
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
    
    # Plot 3: IC function
    ax3 = plt.subplot(1, 3, 3)
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
    
    plt.tight_layout()
    return fig,fig_err


def plot_training_history(history):
    """Plot training loss history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 4))
    
    axes[0,0].semilogy(history['total'], 'k-', linewidth=1.5)
    axes[0,0].set_title('Total Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].semilogy(history['pde'], 'b-', linewidth=1.5)
    axes[0,1].set_title('PDE Residual Loss')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].semilogy(history['ic'], 'r-', linewidth=1.5)
    axes[1,0].set_title('IC Loss')
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
        branch_hidden=cfg.model.branch_hidden,
        trunk_hidden=cfg.model.trunk_hidden,
        p=cfg.model.p,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Train
    history = model.train_pinn(
        n_epochs=cfg.training.n_epochs,
        n_colloc=cfg.training.n_colloc,
        lr=cfg.training.lr,
        theta_gt_range=cfg.training.theta_gt_range,
        theta0_0_range=cfg.training.theta0_0_range
    )
    # Load ground truth data for comparison
    gt_data = np.loadtxt(to_absolute_path('gt_bioheat1D_without_src.csv'))

    #Save model
    cur_dir = os.getcwd()
    print("Current working directory:", cur_dir)

    model_path = os.path.join(cur_dir, "pinn_deeponet_heat_without_src.pth")
    torch.save(model.state_dict(), model_path)
    print("✔ Model saved:", "pinn_deeponet_heat_without_src.pth")

    # Plot results
    print("Generating plots...")
    
    fig1 = plot_training_history(history)
    fig1_path = os.path.join(cur_dir, "pinn_deeponet_training_without_src.png")
    fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
    print("✔ Training plot saved:", "pinn_deeponet_training_without_src.png")
    
    fig2,fig_err = plot_solution(model, theta_gt_test = 0.95238, theta0_0_test = 0.829, gt=gt_data)
    fig2_path = os.path.join(cur_dir, "pinn_deeponet_solution_without_src.png")
    fig_err_path = os.path.join(cur_dir, "comparison.png")
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    fig_err.savefig(fig_err_path, dpi=150, bbox_inches='tight')
    print("✔ Solution plot saved:", "pinn_deeponet_solution_without_src.png")
    print("✔ Comparison plot saved:", "comparison.png")

    print("File saved in:", cur_dir)
    
    plt.show()

if __name__ == "__main__":
    main()

