---
name: Model Structure
---

In this project, we are constructing hydrologic models by combining individual functional zones that represent different parts of the landscape. Hydrologic zones, like `SurfaceZone`, `SnowZone`, and `GroundZone`, act like LEGO pieces that are slotted together. Each zone is represented by one state value `s` representing the water storage in the zone. Each zone may have up to 4 characteristic fluxes of the following:

- `q_forc`: The flux from the forcing data, including precipitation and the water from any zone flowing into this zone
- `q_vap`: The flux representing the vaporization of water, like evapotranspiration
- `q_lat`: The flux moving downslope to either the river or the next zone
- `q_vert`: The flux downwards to the zone below the current one. If there is no zone below, then this must equal zero.

The user combines these zones together to represent a catchment using whatever level of granularity they like. The idea is that the user can start out with a simple model and then add complexity through additional zones when needed. I got the idea for the model when we found that the hydrologic model HBV was limited in its structure and it's inability to represent lateral heterogeneity, like hillslope versus riparian zones, thereby limiting the ability to represent hydrologic connectivity in the landscape.
