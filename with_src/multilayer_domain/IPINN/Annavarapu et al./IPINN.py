"""
I-PINNs for 1D Poisson with THREE interfaces (4 subdomains)
----------------------------------------------------------
Adapts Algorithm 1 + loss functional (MSE_eq + MSE_bc^d + MSE_bc^n + MSE_ic^d + MSE_ic^n)
to the specific 1D Poisson interface problem shown in your figure:

Domain Ω = [0, 1]
Interfaces at x = 0.25, 0.5, 0.75  => 4 non-overlapping subdomains:
  Ω1=[0,0.25], Ω2=[0.25,0.5], Ω3=[0.5,0.75], Ω4=[0.75,1]

Governing equation in each Ωm (m=1..4):
  d/dx( κ_m * du_m/dx ) = f_m      (Poisson with discontinuous κ)

Boundary conditions:
  u1(0) = 0
  u4(1) = 0

Interface conditions at x_i ∈ {0.25,0.5,0.75}:
  [u] = 0                      (continuity)
  [κ du/dx] = 0                (flux continuity)

Coefficients:
  κ1=1, κ2=0.1, κ3=0.5, κ4=0.75
Source:
  f_m = 1  (constant, all regions)

This script trains 4 separate networks (one per subdomain) and enforces coupling via interface losses.
"""

import time
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict

import torch
import torch.nn as nn


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
        layers = [nn.Linear(1, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

        # Step 1: random init (explicit)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,1)
        return self.net(x)


# -----------------------------
# Autodiff helpers (1D)
# -----------------------------
def du_dx(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """u: (N,1), x: (N,1) -> du/dx: (N,1)"""
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]


def d2u_dx2(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """u: (N,1), x: (N,1) -> d2u/dx2: (N,1)"""
    dudx = du_dx(u, x)
    return torch.autograd.grad(
        dudx, x,
        grad_outputs=torch.ones_like(dudx),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]


# -----------------------------
# Problem definition (as in the figure)
# PDE: d/dx(κ du/dx) = f
# For constant κ in each region: κ d2u/dx2 = f  => residual = κ u_xx - f
# -----------------------------
KAPPAS = [1.0, 0.1, 0.5, 0.75]
INTERFACES = [0.25, 0.50, 0.75]
SUBDOMAINS = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0)]


def f_m(x: torch.Tensor) -> torch.Tensor:
    # constant source term f=1
    return torch.ones_like(x)


# Dirichlet boundary conditions
def g_left(x: torch.Tensor) -> torch.Tensor:
    # u(0)=0
    return torch.zeros_like(x)


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
def mse_eq_1d(u_model: nn.Module, x: torch.Tensor, kappa: float, f: Callable) -> torch.Tensor:
    u = u_model(x)
    u_xx = d2u_dx2(u, x)
    res = -kappa * u_xx - f(x)  # κ u_xx - f = 0  (equivalent to d/dx(κ u_x) = f for const κ)
    return mse(res)


def mse_bc_dirichlet(u_model: nn.Module, xb: torch.Tensor, gb: Callable, alpha: float = 1.0) -> torch.Tensor:
    u = u_model(xb)
    return alpha * mse(u - gb(xb))


def mse_interface_continuity(uL: nn.Module, uR: nn.Module, xi: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    # [u] = uL(xi) - uR(xi) = 0
    jump = uL(xi) - uR(xi)
    return alpha * mse(jump)


def mse_interface_flux(uL: nn.Module, uR: nn.Module, xi: torch.Tensor, kL: float, kR: float, alpha: float = 1.0) -> torch.Tensor:
    # [κ u_x] = κL uL_x - κR uR_x = 0
    uL_val = uL(xi)
    uR_val = uR(xi)
    uL_x = du_dx(uL_val, xi)
    uR_x = du_dx(uR_val, xi)
    jump = kL * uL_x - kR * uR_x
    return alpha * mse(jump)


# -----------------------------
# Training (Algorithm 1)
# -----------------------------
@dataclass
class TrainConfig:
    width: int = 64
    depth: int = 4
    lr: float = 1e-3
    max_iters: int = 5000
    print_every: int = 500

    # weights α
    alpha_bc_d: float = 1.0
    alpha_bc_n: float = 0.0   # not used in this specific example
    alpha_int: float = 10.0


def train_ipinn_poisson_3interfaces(cfg: TrainConfig, sampler: Sampler1D):
    device = sampler.device

    # Step 2: NN per subdomain (4 networks)
    nets = [MLP1D(cfg.width, cfg.depth).to(device) for _ in range(4)]
    opt = torch.optim.Adam([p for net in nets for p in net.parameters()], lr=cfg.lr)

    t0 = time.time()
    for it in range(cfg.max_iters):
        # Step 3-4: collocation points + segregation
        x_int = sampler.sample_all_interiors()     # list of 4 tensors
        x0, x1 = sampler.sample_boundaries()       # Dirichlet boundaries
        x_if = sampler.sample_interfaces()         # list of 3 tensors at interfaces

        # Step 5: losses
        # PDE residual
        MSE_eq = torch.zeros((), device=device)
        for m in range(4):
            MSE_eq = MSE_eq + mse_eq_1d(nets[m], x_int[m], KAPPAS[m], f_m)

        # Dirichlet BCs at external boundaries:
        # u1(0)=0 uses net[0] at x0
        # u4(1)=0 uses net[3] at x1
        MSE_bc_d = mse_bc_dirichlet(nets[0], x0, g_left, cfg.alpha_bc_d) + \
                   mse_bc_dirichlet(nets[3], x1, g_right, cfg.alpha_bc_d)

        # No Neumann BC in this example (kept for structure)
        MSE_bc_n = torch.zeros((), device=device)

        # Interface conditions: 3 interfaces, between (1,2), (2,3), (3,4)
        MSE_ic_d = torch.zeros((), device=device)
        MSE_ic_n = torch.zeros((), device=device)

        # interface at 0.25 between Ω1 and Ω2
        MSE_ic_d = MSE_ic_d + mse_interface_continuity(nets[0], nets[1], x_if[0], cfg.alpha_int)
        MSE_ic_n = MSE_ic_n + mse_interface_flux(nets[0], nets[1], x_if[0], KAPPAS[0], KAPPAS[1], cfg.alpha_int)

        # interface at 0.5 between Ω2 and Ω3
        MSE_ic_d = MSE_ic_d + mse_interface_continuity(nets[1], nets[2], x_if[1], cfg.alpha_int)
        MSE_ic_n = MSE_ic_n + mse_interface_flux(nets[1], nets[2], x_if[1], KAPPAS[1], KAPPAS[2], cfg.alpha_int)

        # interface at 0.75 between Ω3 and Ω4
        MSE_ic_d = MSE_ic_d + mse_interface_continuity(nets[2], nets[3], x_if[2], cfg.alpha_int)
        MSE_ic_n = MSE_ic_n + mse_interface_flux(nets[2], nets[3], x_if[2], KAPPAS[2], KAPPAS[3], cfg.alpha_int)

        # Total loss ζ(θ)
        loss = MSE_eq + MSE_bc_d + MSE_bc_n + MSE_ic_d + MSE_ic_n

        # Step 6: update
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (it % cfg.print_every) == 0 or it == cfg.max_iters - 1:
            dt = time.time() - t0
            print(
                f"it={it:6d} | ζ={loss.item():.3e} | "
                f"eq={MSE_eq.item():.3e} | bc_d={MSE_bc_d.item():.3e} | "
                f"ic_d={MSE_ic_d.item():.3e} | ic_n={MSE_ic_n.item():.3e} | t={dt:.1f}s"
            )

    return nets


# -----------------------------
# Prediction (piecewise)
# -----------------------------
@torch.no_grad()
@torch.no_grad()
def predict_piecewise(nets: List[nn.Module], x: torch.Tensor) -> torch.Tensor:
    """
    x: (N,1) in [0,1]
    returns u(x) choosing the correct subdomain net
    """
    assert x.ndim == 2 and x.shape[1] == 1, f"x must be (N,1), got {x.shape}"

    out = torch.zeros_like(x)

    x1d = x[:, 0]  # shape (N,)

    masks = [
        (x1d <= 0.25),
        (x1d > 0.25) & (x1d <= 0.50),
        (x1d > 0.50) & (x1d <= 0.75),
        (x1d > 0.75),
    ]

    for m, mask in enumerate(masks):
        if mask.any():
            out[mask, 0:1] = nets[m](x[mask, 0:1])

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
        max_iters=4000,
        print_every=500,
        alpha_bc_d=1.0,
        alpha_int=20.0,  # often helps for strong enforcement of jumps
    )

    nets = train_ipinn_poisson_3interfaces(cfg, sampler)

    #======Plots============
    import matplotlib.pyplot as plt

    # evaluation on grid
    xg = torch.linspace(0, 1, 1001, device=device).reshape(-1, 1)
    ug = predict_piecewise(nets, xg)

    # move to cpu for plotting
    xg_np = xg.detach().cpu().numpy().flatten()
    ug_np = ug.detach().cpu().numpy().flatten()

    # plot
    plt.figure(figsize=(8, 4))
    plt.plot(xg_np, ug_np, label="I-PINN solution", linewidth=2)

    # mark interfaces
    for xi in [0.25, 0.5, 0.75]:
        plt.axvline(x=xi, color="k", linestyle="--", alpha=0.5)

    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Poisson equation with three interfaces – I-PINN solution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()