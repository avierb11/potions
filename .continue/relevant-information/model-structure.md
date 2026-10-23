---
name: Model Structure
---

In this project, we are constructing hydrologic models by combining individual functional zones that represent different parts of the landscape. Hydrologic zones, like `SurfaceZone`, `SnowZone`, and `GroundZone`, act like LEGO pieces that are slotted together. Each zone is represented by one state value `s` representing the water storage in the zone. Each zone may have up to 4 characteristic fluxes of the following:

- `q_forc`: The flux from the forcing data, including precipitation and the water from any zone flowing into this zone
- `q_vap`: The flux representing the vaporization of water, like evapotranspiration
- `q_lat`: The flux moving downslope to either the river or the next zone
- `q_vert`: The flux downwards to the zone below the current one. If there is no zone below, then this must equal zero.

The user combines these zones together to represent a catchment using whatever level of granularity they like. The idea is that the user can start out with a simple model and then add complexity through additional zones when needed. I got the idea for the model when we found that the hydrologic model HBV was limited in its structure and it's inability to represent lateral heterogeneity, like hillslope versus riparian zones, thereby limiting the ability to represent hydrologic connectivity in the landscape.

## Area fraction convention (intensive fluxes, extensive boundaries)

Each surface zone carries an area fraction (`scales`, which must be strictly
positive and sum to 1 across *all* surface zones of *all* hillslopes). Area
factors live on the **boundaries** of the connectivity graph, never inside a
zone's flux equations:

- **Forcing into a zone `i`** is weighted by its area fraction `s_i`
  (`get_forc_mat`). A zone that covers 5% of the catchment receives 5% of the
  precipitation and PET as an *intensive depth* (mm/d).
- **A lateral edge from donor `d` to receiver `r`** carries the area ratio
  `s_d / s_r` (`lat_mat_scaled`). This concentrates all the runoff leaving the
  donor's area onto the receiver's (usually smaller) area and conserves mass.
  A hillslope sending `q` mm/d to a riparian zone 1/19th its size delivers
  `19q` mm/d to that zone.
- **The river-outflow row** carries the receiver scale `s_r`, so summing the
  river zones' lateral fluxes yields a catchment-mean depth (mm/d) matching the
  measured streamflow (discharge / catchment area).
- **Zone fluxes stay intensive (mm/d).** A vertical edge within a column joins
  two zones of equal area, so its ratio is 1 and no adjustment is needed. When
  a layer *aggregates* (N zones above flow into 1 zone below), the receiving
  zone represents the combined area, so the vertical edge carries the ratio
  `s_d / s_r` (`vert_mat_scaled`) to conserve mass.

The plain `lat_mat` (0/1 adjacency) is still used by the reactive-transport
coupling, which mixes *concentrations* using water fluxes that are already
area-weighted; only the hydro step and the outlet use `lat_mat_scaled` and
`vert_mat_scaled`.

The practical consequence: a zone's parameters (melt factor, capacities,
recession coefficients) describe its per-unit-area response and are therefore
scale-independent. Changing `s_i` no longer forces the calibration to absorb
zone size.

