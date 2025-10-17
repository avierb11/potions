# Reactive Transport
The reactive transport model, like the hydrologic model, can be simplified down into a series of buckets that all can be solved independently. Luckily, all of the zones are identical, meaning that you don't have to decide which type of zone to use where - you can just use a single zone and specify different parameters for each.

## Processes in each zone
Within a single reactive transport zone, the two main processes are right in the name:
- **Reaction**: Alteration of existing species within each zone, typically dissolution or equilibrium
- **Transport**: Movement of mobile species between zones

So, the overall relationship describing the flow of mass through the system is:
$$ 
\frac{dm_i}{dt} = \left(\frac{dm_i}{dt}\right)_{\text{reaction}} + \left(\frac{dm_i}{dt}\right)_{\text{transport}}
$$

Where $m_i$ is the mass of a single chemical species within a zone. This only changes due to reaction and transport. Anything else is simplified out of this model. Now, this is a simple ODE that we can solve directly using `scipy`, and it just comes down to breaking down these equations into a form that we can input it into a computer.

## Reactions
Kinetic chemical reactions are fairly complex, but we simplify this down to just two main types of kinetic reactions:
- **Transition State Theory (TST) Reactions**: Describing mineral reactions
- **Monod Reactions**: Describing organic matter reactions

In addition to those formats, there are other environmental drivers like temperature, soil moisture, and proximity to the water table that all have their own modifications. In short, there are really _four_ types of functions involved, and the overall rate of reaction may look something like:
$$
\frac{dm}{dt} = V(t) \theta_{ssa} r_i f(C(t)) f(S_w) f(Z_w) f(T)
$$
Where:
- $V(t)$: The volume of water in the zone
- $\theta_{ssa}$: The specific surface area of the mineral
- $r_i$: The kinetic rate constant of the mineral
- $f(C(t))$ The portion of the reaction dependent on concentrations - this is where the difference between TST and Monod reactions shows up
- $f(S_w)$: The portion of the reaction dependent on soil moisture
- $f(Z_w)$: The portion of the reaction dependent on water table depth
- $f(T)$: The portion of the reaction dependent on temperature

Put altogether, everything but $f(C(t))$ can be calculated as a vector quite easily. Each species in each zone may have its own parameters in the auxiliary functions (everything but $f(C(t))$), but this is simple. With each species having it's own reaction characteristics, potentially being a TST or Monod rate, we really need a way to cleanly solve for $f(C(t))$ without assuming a rate constant, while also keeping in mind that the rate for a single species is a vector-valued function; it requires knowing the concentrations of all species.

One way around this problem is to just decompose $f(C(t))$ into two components: $f(C(t))_{TST}$ and $f(C(t))_{Monod}$. In this way, the problem becomes much simpler:
$$
f(C(t)) = f(C(t))_{TST} + f(C(t))_{Monod}
$$

Now, we can just assume that there is a TST and a Monod component for each species. In reality, it's either one or the other, but we can just set the parameters to assume a zero rate the one that a species is not.

### TST Reactions
For a TST rate law, the equation is:
$$
f(C(t))_{i, TST} = d(C(t))(1 - \frac{1}{K_{eq, i}} IAP_i(t))
$$
Where 
- $d(C(t))$: A function that describes how the reaction may be dependent on other species, like pH. This function is like taking $C(t)$ to the power of $\vec{d}$ element-wise and adding all the terms up. Not every mineral has dependence terms, they just sometimes do.
    - $d_i(C(t)) = \sum_{i=1}^{n} C_i(t)^{\vec{d_i}}$
- $K_{eq,i}$ The equilibrium constant for the mineral
- $IAP(t)$ is the ion activity product describing how far the term is from equilibrium. This is just broken down into log transforms: $\ln (IAP_i(t)) = \nu_i \ln(C_i(t))$.
    - $\nu_i$: The vector describing the stoichiometry of the reaction of the mineral. This is purely from the database.


Put in vector form, these can be simplified fairly easily, with $\mathbf{D}$ as the dependence vector, with dimensions (num minerals, num primary aqueous species), and $\mathbf{N}$ as the stoichiometry matrix with the same dimensions. The dependence and ion activity product terms can easily be calculated using log-transforms to make the problems linear:

$$
\ln (IAP(t)) = \mathbf{N} \ln(C(t))
$$

$$
\ln d = \mathbf{D} \ln(C(t))
$$

### Monod Reactions
For a Monod rate law, the equation is
$$
f(C(t))_{i, Monod} = \left( \prod_{j=1}^{n}\frac{C_j}{k_j + C_j} \right) \left(  \prod_{j=1}^{n} \frac{i_j}{i_j + C_j} \right)
$$
In short, this just comes down to some easy code in Python that's not even worth looking at outside of the code base.