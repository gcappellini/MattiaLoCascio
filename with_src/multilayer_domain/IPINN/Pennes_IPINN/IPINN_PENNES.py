"""
I-PINNs for 1D Pennes with ONE interfaces (2 subdomains)
----------------------------------------------------------
Adapts Algorithm 1 + loss functional (MSE_eq + MSE_bc^d + MSE_bc^n + MSE_ic^d + MSE_ic^n)
to the specific 1D Poisson interface problem shown in your figure:

Domain Ω = [0, 1]
Interfaces at x = 0.5  => 2 non-overlapping subdomains:
Ω1=[0.0,0.5], Ω2=[0.5,1.0]

Governing equation in each Ωm (m=1..4):
  a1 * du_dt = du2_dx2 + wb * a2 * u - src_values     (Pennes with discontinuous κ)

Boundary conditions:
y3(t) = 0.65*(1-exp(-t/0.5))
  du1_dx(0) = -a5*(y3(t)-u)
  u2(1) = 0

Interface conditions at x_i ∈ {0.5}:
  [u] = 0                      (continuity)
  [du/dx] = 0                (flux continuity)

Coefficients:
    a1= [27.72760159 20.71565275]
    a2= [49910.65333333 37288.9]
    a3= [2.06635779 1.05958758]
    a4= [0.34956447 0.34956447]
    a5= 175.0
Source:
  f_m = P *a3* exp(-a4*x)  (constant, all regions)

This script trains 4 separate networks (one per subdomain) and enforces coupling via interface losses.
"""

import time
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import numpy as np


# -----------------------------
# Basics
# -----------------------------
def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mse(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x**2)


# -----------------------------
# 1D MLP per subdomain
# -----------------------------
class MLP1D(nn.Module):
    def __init__(self, width: int = 64, depth: int = 4, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(2, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers) #shape (N,1)

        # Step 1: random init (explicit)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xt):
        # xt: (N,2)
        x = xt[:,0:1]
        t = xt[:,1:2]
        bc_factor = t * (x-1) # u(x,0)=0 and u(1,t)=0
        return bc_factor*self.net(xt) 


# -----------------------------
# Autodiff helpers (1D)
# -----------------------------

def du_dt(u: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    """u: (N,1), x: (N,1) -> du/dx: (N,1)"""
     # Enable gradients for xt
    return torch.autograd.grad(
        u, xt,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0][:,1]

def du_dx(u: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    """u: (N,1), x: (N,1) -> du/dx: (N,1)"""
     # Enable gradients for xt
    return torch.autograd.grad(
        u, xt,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0][:,0]

def d2u_dx2(u: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    """u: (N,1), x: (N,1) -> d2u/dx2: (N,1)"""
    dudx = du_dx(u, xt)
    return torch.autograd.grad(
        dudx, xt,
        grad_outputs=torch.ones_like(dudx),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0][:,0]

# -----------------------------
# Problem definition (as in the figure)
# PDE: d/dx(κ du/dx) = f
# For constant κ in each region: κ d2u/dx2 = f  => residual = κ u_xx - f
# -----------------------------
a1= [27.72760159,20.71565275]
a2= [49910.65333333,37288.9]
a3= [2.06635779,1.05958758]
a4= [0.34956447,0.34956447]
a5= 175.0
KAPPAS = [0.21,0.49]
INTERFACES = [0.50]
SUBDOMAINS = [(0.0, 0.50), (0.50, 1.0)]

# Source term f_m = P * a3 * exp(-a4*x)
def f_m(x: torch.Tensor, P: float, a3: float, a4: float) -> torch.Tensor:
    src = P * a3 * torch.exp(-a4*x)
    return src

# Robin boundary condition
def g_left(x0t: torch.Tensor, a5: float, u_model: nn.Module) -> torch.Tensor:
    # du_dx(0,t)=a5*(y3(t)-u(0,t))
    t = x0t[:,1:2]
    y3_t = 0.65 * (1 - torch.exp(-t / 0.5))
    u0 = u_model(x0t)
    return a5*(y3_t-u0)

# Dirichlet boundary condition
# def g_left(x: torch.Tensor) -> torch.Tensor:
#     # u(0)=0
#     return torch.zeros_like(x)

def g_right(x: torch.Tensor) -> torch.Tensor:
    # u(1)=0
    return torch.zeros_like(x)

# -----------------------------
# Sampling (collocation points)
# -----------------------------
@dataclass
class Sampler1D:
    N_eq: int
    N_bc: int
    N_int: int
    device: torch.device
    dtype: torch.dtype = torch.float32

    def sample_interior_subdomain(self, a: float, b: float, N: int) -> torch.Tensor:
        # interior points uniform in (a,b)
        x = a + (b - a) * torch.rand((N, 1), device=self.device, dtype=self.dtype)
        x.requires_grad_(True)
        return x

    def sample_all_interiors(self) -> List[torch.Tensor]:
        xs = []
        for (a, b) in SUBDOMAINS:
            xs.append(self.sample_interior_subdomain(a, b, self.N_eq))
        return xs

    def sample_boundaries(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # points at x=0 and x=1 (Dirichlet)
        x0 = torch.zeros((self.N_bc, 1), device=self.device, dtype=self.dtype)
        x1 = torch.ones((self.N_bc, 1), device=self.device, dtype=self.dtype)
        x0.requires_grad_(True)
        x1.requires_grad_(True)
        return x0, x1

    def sample_interfaces(self) -> List[torch.Tensor]:
        # replicate interface points (in 1D it's just the coordinate, but we use N_int copies for stable training)
        xis = []
        for xi in INTERFACES:
            x = torch.full((self.N_int, 1), float(xi), device=self.device, dtype=self.dtype)
            x.requires_grad_(True)
            xis.append(x)
        return xis
    
# -----------------------------
# Loss terms (matching the paper structure)
# ζ = MSE_eq + MSE_bc^d + MSE_bc^n + MSE_ic^d + MSE_ic^n
#
# Here:
# - MSE_bc^n is not used because the problem in the figure uses ONLY Dirichlet at external boundaries.
#   We'll keep the term for completeness (it will be zero), so the structure matches Algorithm 1.
# -----------------------------
def mse_eq_1d(u_model: nn.Module, xt: torch.Tensor, a1: float, a2:float, src_values: torch.Tensor) -> torch.Tensor:
    wb=0.005
    xt_requires_grad = xt.clone().requires_grad_(True)
    u = u_model(xt_requires_grad)          #(N,1)
    u_t = du_dt(u, xt_requires_grad)       #(N,1)
    u_xx = d2u_dx2(u, xt_requires_grad)    #(N,1)
    res = a1 * u_t - u_xx + wb * a2 * u - src_values  #(N,1)
    return mse(res)


def mse_bc_robin(u_model: nn.Module, x0t: torch.Tensor, g_left: Callable, a5:float) -> torch.Tensor:
    u = u_model(x0t)
    return mse(u - g_left(x0t, a5, u_model))

def mse_bc_dirichlet(u_model: nn.Module, x1t: torch.Tensor, g_right: Callable) -> torch.Tensor:
    u = u_model(x1t)
    x1 = x1t[:,0:1]
    return mse(u - g_right(x1))


def mse_interface_continuity(uL: nn.Module, uR: nn.Module, xit: torch.Tensor) -> torch.Tensor:
    # [u] = uL(xi) - uR(xi) = 0
    jump = uL(xit) - uR(xit)
    return mse(jump)


def mse_interface_flux(uL: nn.Module, uR: nn.Module, xit: torch.Tensor, kL: float, kR: float) -> torch.Tensor:
    # [κ u_x] = κL uL_x - κR uR_x = 0
    xit_requires_grad = xit.clone().requires_grad_(True)
    uL_val = uL(xit_requires_grad)
    uR_val = uR(xit_requires_grad)
    uL_x = du_dx(uL_val, xit_requires_grad)
    uR_x = du_dx(uR_val, xit_requires_grad)
    jump = kL * uL_x - kR * uR_x
    return mse(jump)



# -----------------------------
# Training (Algorithm 1)
# -----------------------------
@dataclass
class TrainConfig:
    width: int = 64
    depth: int = 4
    lr: float = 1e-3
    max_iters: int = 10000
    print_every: int = 500

    # # weights α
    # #alpha_bc_d: float = 1.0
    # alpha_bc_r: float = 1.0
    # alpha_int: float = 10.0


def train_ipinn_pennes_one_interface(cfg: TrainConfig, sampler: Sampler1D):
    device = sampler.device

    # Step 2: NN per subdomain (4 networks)
    nets = [MLP1D(cfg.width, cfg.depth).to(device) for _ in range(4)]
    opt = torch.optim.Adam([p for net in nets for p in net.parameters()], lr=cfg.lr)

    t0 = time.time()
    for it in range(cfg.max_iters):
        # Step 3-4: collocation points + segregation
        x_int = sampler.sample_all_interiors()     # list of 2 tensors
        x0, x1 = sampler.sample_boundaries()       # Dirichlet and Robin boundaries
        x_if = sampler.sample_interfaces()         # list of 1 tensor at interface

        # Step 5: losses
        # PDE residual
        t = torch.linspace(0,1,sampler.N_eq,device=device).reshape(-1,1)  #(N_eq,1)
        MSE_eq = torch.zeros((), device=device)
        for m in range(2):
            x = x_int[m]   #(Neq,1)
            xt = torch.cat([x, t], dim=1)  #(Neq,2)
            src_values = f_m(x, P=125.0, a3=a3[m], a4=a4[m])  #(Neq,1)
            MSE_eq = MSE_eq + mse_eq_1d(nets[m], xt, a1[m], a2[m], src_values)

        # Dirichlet BCs at external boundaries:
        # u1(0)=0 uses net[0] at x0
        # u4(1)=0 uses net[3] at x1
        t = torch.linspace(0,1,sampler.N_bc,device=device).reshape(-1,1)  #(N_bc,1)
        x0t = torch.cat([x0, t], dim=1)  #(N_bc,2)
        x1t = torch.cat([x1, t], dim=1)  #(N_bc,2)
        # MSE_bc_d = mse_bc_dirichlet(nets[0], x0t, g_left) + \
        #             mse_bc_dirichlet(nets[1], x1t, g_right)#, cfg.alpha_bc_d)
        MSE_bc_r = mse_bc_robin(nets[0], x0t, g_left, a5)#, cfg.alpha_bc_d)

        # Interface conditions: 1 interface between (1,2)
        MSE_ic_d = torch.zeros((), device=device)
        MSE_ic_n = torch.zeros((), device=device)

        # interface at 0.25 between Ω1 and Ω2
        t_interface = torch.linspace(0,1,sampler.N_int,device=device).reshape(-1,1)  #(N_int,1)
        xit = torch.cat([x_if[0], t_interface], dim=1)  #(N_int,2)
        MSE_ic_d = MSE_ic_d + mse_interface_continuity(nets[0], nets[1], xit)#, cfg.alpha_int)
        MSE_ic_n = MSE_ic_n + mse_interface_flux(nets[0], nets[1], xit, KAPPAS[0], KAPPAS[1])#, cfg.alpha_int)

        #Weights
        if it == 0:
            initial_losses = torch.tensor([MSE_eq.item(), MSE_bc_r.item(), MSE_ic_d.item(), MSE_ic_n.item()], device=device)
            w_eq = 5.0 / (initial_losses[0])
            w_bc_r = 5.0 / (initial_losses[1])
            w_ic_d = 5.0 / (initial_losses[2])
            w_ic_n = 5.0 / (initial_losses[3])

        # Total loss ζ(θ)
        loss = w_eq * MSE_eq + w_bc_r * MSE_bc_r + w_ic_d * MSE_ic_d + w_ic_n *MSE_ic_n

        # Step 6: update
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (it % cfg.print_every) == 0 or it == cfg.max_iters - 1:
            dt = time.time() - t0
            print(
                f"it={it:6d} | ζ={loss.item():.6f} | "
                f"eq={MSE_eq.item():.6f} | bc_r={MSE_bc_r.item():.6f} | "
                f"ic_d={MSE_ic_d.item():.6f} | ic_n={MSE_ic_n.item():.6f} | t={dt:.1f}s"
            )

    return nets

# -----------------------------
# Prediction (piecewise)
# -----------------------------
@torch.no_grad()
@torch.no_grad()
def predict_piecewise(nets: List[nn.Module], xt: torch.Tensor) -> torch.Tensor:
    assert xt.ndim == 2 and xt.shape[1] == 2, f"xt must be (N,2), got {xt.shape}"

    out = torch.zeros((xt.shape[0], 1), device=xt.device)

    x = xt[:, 0]   # (N,) 

    masks = [
        x <= 0.50,
        x >  0.50
    ]

    for m, mask in enumerate(masks):
        if mask.any():
            out[mask] = nets[m](xt[mask])

    return out

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sampler = Sampler1D(
        N_eq=512,     # interior per subdomain
        N_bc=64,      # boundary points at x=0 and x=1
        N_int=128,    # "copies" at each interface (stabilizes training)
        device=device
    )

    cfg = TrainConfig(
        width=64,
        depth=4,
        lr=1e-3,
        max_iters=5000,
        print_every=500
        # alpha_bc_d=1.0,
        # alpha_int=20.0,  # often helps for strong enforcement of jumps
    )

    nets = train_ipinn_pennes_one_interface(cfg, sampler)


    #======Plots============
    import matplotlib.pyplot as plt
    
    # evaluation on grid
    nx = 101
    nt = 101
    gt = np.loadtxt('gt_bioheat1D_src_multilayer.csv')
    u_gt = gt[:, 2].reshape(nx, nt, order='F')

    x_plot = torch.linspace(0, 1, nx, device=device)
    t_plot = torch.linspace(0, 1, nt, device=device)
    time_indices = [0, nt//4, nt//2, 3*nt//4, nt-1]
    
    # Create mesh grid for evaluation
    x_mesh, t_mesh = torch.meshgrid(x_plot, t_plot, indexing='ij')
    xtg = torch.stack([x_mesh, t_mesh], dim=-1).reshape(-1, 2)
    u_pred = predict_piecewise(nets, xtg).reshape(nx, nt)

    plt.figure(figsize=(10, 6))
    for i, idx in enumerate(time_indices):
        t_val = t_plot[idx].item()
        color = plt.cm.tab10(i)

        plt.plot(x_plot.cpu().numpy(), u_pred[:, idx].detach().cpu().numpy(), label=f'I-PINN t={t_val:.3f}', color=color)
        plt.plot(x_plot.cpu().numpy(), u_gt[:, idx], '--', label=f'GT t={t_val:.3f}', linewidth=1, color=color)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x,t)', fontsize=12)
    plt.title('Pennes equation with one interface, I-PINN vs Ground truth', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig = True
    if save_fig:
        plt.savefig('pennes_1D_one_interface_ipinn_vs_gt.png', dpi=300)
    plt.axvline(0.5, color='gray', linestyle='--', linewidth=2)  # Interface point
    plt.show()
