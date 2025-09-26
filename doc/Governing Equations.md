# Governing equations
The following equations describe the change in concentration of a chemical species within a single, well-mixed control volume, often referred to as a "bucket" model. This is a common approach for lumped-parameter hydrologic and biogeochemical models.

## Advective Transport

### Symbols
- $m(t)$: the number of moles of some species in a bucket at any time
- $V(t)$: the volume of water in a bucket at any time
- $C(t)$: the concentration of the species
- $Q_{in}$: rate of water flux into the bucket
- $Q_{out}$: rate of water flux out of the bucket
- $\Delta Q = Q_{in} - Q_{out}$: Rate of change of water volume in the bucket

### Internal Mass Balance
First, we can express the rate of change of mass, $m'(t)$, by applying the product rule to the definition of mass, $m(t) = C(t)V(t)$:
$$
\begin{aligned}
m(t) &= C(t) V(t) \\
m'(t) &= C(t) V'(t) + C'(t) V(t) \\
&= C(t)(Q_{in} - Q_{out}) + C'(t) V(t) \\
&= \Delta Q C(t) + C'(t) V(t) \\
\end{aligned}
$$
### Flux mass balance
Second, we can express the rate of change of mass based on the fluxes into and out of the bucket. This assumes a well-mixed system, where the concentration of the water leaving the bucket, $C(t)$, is the same as the concentration within the bucket.
$$
\begin{aligned}
m'(t) &= \dot{m}_{in} - \dot{m}_{out} \\
&= C_{in}Q_{in} - C(t) Q_{out}
\end{aligned}
$$

### Combining the mass balance expressions
By equating these two expressions for the rate of change of mass, $m'(t)$, we can solve for the ordinary differential equation (ODE) that governs the change in concentration, $C'(t)$.
$$
\begin{aligned}
\Delta Q C(t) + C'(t) V(t) &= C_{in}Q_{in} - C(t) Q_{out} \\
C'(t) V(t) &= C_{in}Q_{in} - C(t) Q_{out} - \Delta Q C(t) \\
C'(t) V(t) &= C_{in}Q_{in} - C(t) Q_{out} - Q_{in} C(t) + Q_{out} C(t) \\
C'(t) V(t) &= C_{in}Q_{in} - \cancel{C(t) Q_{out}} - Q_{in} C(t) + \cancel{Q_{out} C(t)} \\
C'(t) V(t) &= C_{in}Q_{in} - Q_{in} C(t) \\
C'(t) V(t) &= Q_{in}(C_{in} - C(t)) \\
C'(t) &= \frac{Q_{in}(C_{in} - C(t))}{V(t)} \\
&= \frac{Q_{in}(C_{in} - C(t))}{V_0 + \Delta Q \Delta t} \\
C'(t) &= \frac{Q_{in}}{V_0 + \Delta Q \Delta t}(C_{in} - C(t)) \\
\end{aligned}
$$

This final equation is the governing ODE for a completely mixed reactor. It states that the rate of change in concentration is driven by the difference between the incoming concentration and the current internal concentration, scaled by the ratio of the inflow rate to the current volume of water in the bucket.

---
## Biogeochemical Reactions
The change in chemical concentrations due to reactions is often modeled by splitting the problem into two parts:
1.  **Speciation:** Fast, equilibrium-based reactions that determine the distribution of aqueous species.
2.  **Kinetics:** Slower, rate-limited reactions, such as mineral dissolution or precipitation.

### Speciation
There are 3 types of equations:
1. Equilibrium constants
2. Total concentrations
3. Charge balance

The relevant dimensions in this problem are:
- $n_p$: Number of primary species
- $n_s$: Number of secondary species
- $n_m$: Number of mineral species
- $n_t = n_p + n_m$: Number of total species
- $n_{aq} = n_p + n_s$: Number of aqueous species

#### 1. Equilibrium constants
An equilibrium constant for a secondary species, which is formed from a set of primary species, can be expressed as:
$$
K_i = \prod_{j=1}^{n}C_j^{\nu_{ij}}
$$
Or, more naturally in log scale:
$$
\log K_i = \sum_{j=1}^{n}\nu_{ij}\log C_j
$$

#### 2. Total Concentrations
The total concentration of a primary species is the sum of its free form plus its concentration in all secondary species. The mass-conservation equation for a single primary species is:
$$
C_{t,i} = \sum_{j=1}^{n} \tau_{ij} C_j
$$

#### 3. Charge Conservation
The charge conservation equation is a simple sum:
$$
0 = \sum_{j=1}^{n}z_j C_j
$$

#### Combining the equations
The speciation problem can be expressed as two coupled systems of equations:
##### Log Space
$$
\log \vec{K}: \mathbb{R}^{n_{aq}} \to \mathbb{R}^{n_s} = \mathbf{N} \log\vec{C}
$$
This describes the equilibrium relationships between the primary species and the secondary species. All species in this formula are **aqueous**.

##### Linear Space
$$
\vec{C}_t: \mathbb{R}^{n_{aq}} \to \mathbb{R}^{n_p} = \mathbf{T} \vec{C}
$$


A common and robust method to solve this system is to first find a general solution for the equilibrium equations, and then use that solution to solve the mass balance equations. The equilibrium system is under-determined, and its general solution can be expressed using a particular solution and the null space of the stoichiometric matrix $\mathbf{N}$:

$$
\log \vec{K} = \mathbf{N} \log\vec{C}
$$

So, the solution to this problem will be:
$$
\log \vec{C} = \log \vec{C}_p + \text{null}(\mathbf{N})\log \vec{x}
$$
Where:
- $\log{C}_p = (\mathbf{N}^T\mathbf{N})^{-1}\mathbf{N}^T \log{K}$: a particular solution to the problem obtained by the least squares solution
- $\text{null}(\mathbf{N})$: the null-space of the matrix $\mathbf{N}$
- $\vec{x} \in \mathbf{R}^{n_p}$: the vector of primary unknowns. The degrees of freedom in the system correspond to the primary species, so we can parameterize the solution in terms of these unknowns. Often, $\vec{x}$ is chosen to be the vector of the log-concentrations of the primary species.

**Why did we choose this equation first?**
By parameterizing the solution in log-space, we ensure that any real-valued solution for $\vec{x}$ will result in positive concentrations, which is a critical physical constraint.
$$
\forall \vec{x} \in \mathbb{R}^{n_p}, \vec{C}(\vec{x}) > 0
$$
The null space and the particular solution can be pre-computed, simplifying the problem at each time step.

The actual solution then is:
$$
\vec{C}(\vec{x}) = \exp \left( \log \vec{C}_p + \text{null}(\mathbf{N})\vec{x} \right) \tag{1}
$$
And thus we have a parameterized solution to the concentration. The remaining problem is to find the primary species concentrations:
$$
\begin{aligned}
\vec{C}_t &= \mathbf{T} \vec{C}(\vec{x}) \\
\vec{R}(\vec{x}) &= \mathbf{T} \vec{C}(\vec{x}) - \vec{C}_t \\
\end{aligned}
$$
This is a system of non-linear equations where the residual $\vec{R}(\vec{x})$ must be driven to zero. This can be solved for $\vec{x}$ using a numerical root-finding algorithm (e.g., Newton's method). Once $\vec{x}$ is found, the full set of aqueous concentrations can be calculated using equation $(1)$.

---
### Mineral Reactions
This section outlines the kinetic rate laws for reactions that are too slow to be treated with an equilibrium assumption, such as mineral dissolution and precipitation. The general form for the rate of change of the moles of minerals is:
$$
\frac{d\vec{m}}{dt} = \vec{f}(\vec{C}) \vec{g}(T, S_w, Z_w)
$$
Where:
- $\vec{f} = \langle f_1, ..., f_{n_m} \rangle$: the characteristic functions for the dependence of the mineral rate on the aqueous species concentraions
- $\vec{g}$: a function that lumps together other dependencies, such as on temperature ($T$), soil water content ($S_w$), and water table depth ($Z_w$).

#### Characteristic Equations
The characteristic equations, $f_i(\vec{C})$, describe how the reaction rate depends on the concentrations of aqueous species. Different formulations can be used to represent different conceptual models (e.g., microbially-mediated vs. abiotic reactions). Below are two common examples.

##### 1. Monod-type Kinetics
This form is often used for enzyme or microbially-mediated reactions. It includes terms for species that promote the reaction and species that inhibit it.
$$
f_i(\vec{C}) = \left( \prod_{j=1}^{n}\frac{C_j}{k_j + C_j} \right) \left(  \prod_{j=1}^{n} \frac{i_j}{i_j + C_j} \right)
$$
Where $k_j$ is the half-saturation constant for a promoting species and $i_j$ is the inhibition constant for an inhibiting species.

##### 2. Transition-State Theory (TST) Kinetics
This form is commonly used for abiotic mineral dissolution/precipitation. The rate depends on promoting/inhibiting species and the saturation state of the solution with respect to the mineral.
$$
\begin{aligned}
f_i(\vec{C}) = \left( \prod_{j=1}^{n} C_j^{d_{i,j}} \right) \left(1 - \frac{1}{K_{eq,i}} \prod_{j=1}^{n} C_j^{\nu_j} \right)
\end{aligned}
$$
Where:
- The first term represents catalysis or inhibition by aqueous species (e.g., H+).
- The second term represents the thermodynamic driving force of the reaction, where the product of concentrations is the Ion Activity Product (IAP) and $K_{eq,i}$ is the equilibrium constant for the mineral reaction. The term in parentheses is often written as $(1 - \Omega_i)$, where $\Omega_i$ is the saturation index.

### Coupling Kinetics to Speciation
The kinetic reactions (slow) and equilibrium reactions (fast) are coupled. The dissolution or precipitation of minerals changes the total concentration of primary species in the aqueous phase. This change in total concentration then forces a re-speciation of the aqueous phase. The link is given by:
$$
\frac{dC_{t,i}}{dt} = -\frac{1}{V} \sum_{k=1}^{n_m} \sigma_{ik} \frac{dm_k}{dt}
$$
Where:
- $\frac{dC_{t,i}}{dt}$ is the rate of change of the total concentration of primary species $i$ due to mineral reactions.
- $V$ is the volume of water.
- $\sigma_{ik}$ is the stoichiometric coefficient of primary species $i$ in mineral $k$.
- $\frac{dm_k}{dt}$ is the kinetic rate of change of mineral $k$ from the equations above.

This change is then added to the change from advective transport to get the total rate of change for $\vec{C}_t$, which is then used to solve the speciation problem at the next time step.