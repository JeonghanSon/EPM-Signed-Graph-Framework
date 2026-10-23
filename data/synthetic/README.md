# Synthetic networks

This directory contains the generated signed networks used in the three
synthetic validation experiments reported in the paper. All experiments use
1,000 nodes, eight equally sized ground-truth communities, and graph seeds
0--4.

- `eq1_signed_exact_n1000_k8/`: joint variation of the within-community
  positive-edge fraction and the between-community negative-edge fraction.
- `eq1_antagonistic20_n1000_k8/`: variation of negative-edge placement while
  holding the positive topology and the number of negative edges fixed.
- `negative_prevalence_inter095_with_zero_n1000_k8/`: nested negative-edge
  sets used to vary the negative-edge proportion while holding the positive
  layer fixed.

Each generated graph directory includes its signed edge lists, node/community
metadata, and a `manifest.json` recording the realized construction
parameters and graph fingerprint. The corresponding generation code is under
`src/signed_epm/synthetic/`.
