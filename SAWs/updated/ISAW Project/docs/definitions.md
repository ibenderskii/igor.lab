# 3D_ISAW_PNIPAM_contact_motif_collapse_analysis — frozen definitions

> Generated from `project_definitions.json` by `isaw_schema.render_definitions_md()`. Do not edit by hand; edit the JSON and regenerate (`python isaw_schema.py`).

- schema_version: 2
- definitions_version: 1.1.0
- polymer_system: PNIPAM
- transition_type: LCST

## Chain-length convention
- N = n_beads; n_steps = N - 1.

## Reduced bias, K, q
- b(T) is the reduced contact bias; sampling weight P(C|T) ∝ exp[-m(C) b(T)].
- K(T) = -b(T); thus P(C|T) ∝ exp[K(T) m(C)] and HIGHER K FAVORS MORE CONTACTS.
- q(T) = exp(K(T)) = exp(-b(T)); K is authoritative because q can overflow.
- u(C,T) = m(C) b(T) = -m(C) K(T).
- H(C,T) = T u(C,T); model-implied, NOT automatically a temperature-independent physical energy for polynomial effective models.

## PNIPAM LCST expectations (verify against the fitted model; do not hardcode)
- K(T) increases as T increases
- mean contact count m increases as T increases
- Rg^2 decreases as T increases
- long-range/global contacts and connected contact structure increase as collapse develops
- primary contact-favoring coordinate: K (not temperature alone)

## Fixed contour bins
- short_fixed: 3 <= r <= 9
- medium_fixed: 11 <= r < long_threshold_fixed
- long_fixed: r >= 15
- constraints: 3 <= short_min <= short_max < medium_min < long_threshold < n_beads

## Scaled contour bins
- local_scaled r/N<=0.10; mesoscopic_scaled 0.10<r/N<0.33; global_scaled r/N>=0.33
- constraints: 0 <= local_max_ratio < meso_max_ratio <= 1

## Contact / augmented graph
- contact graph: contact_graph_edges == m and sum_component_edges == m
- augmented graph identities: augmented_graph_edges = N-1+m; augmented_graph_components = 1; augmented_graph_cycle_rank = m

## Primary key
- ('run_id', 'seed', 'snapshot_index', 'temperature_index')

## Required run metadata fields
- run_id, seed, n_beads, n_steps, model_name, param_names, model_params, Tref, Tscale, temperatures
