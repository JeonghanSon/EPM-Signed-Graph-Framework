# Datasets

The experiments use public signed networks from the
[Stanford Network Analysis Project (SNAP)](https://snap.stanford.edu/data/index.html):

- `soc-sign-bitcoin-alpha`
- `soc-sign-bitcoin-otc`
- `wiki-Elec`
- `wiki-RfA`
- `soc-sign-Slashdot090221`
- `soc-sign-epinions`

For a lightweight default run, this repository includes
`data/raw/soc-sign-bitcoinalpha.csv`. Download the other datasets from SNAP and
place them under `data/raw/` using the names listed in
`data/metadata/raw_files.json`.

Preprocessing preserves the transductive node set, restricts the experiment to
the largest connected component, and reserves a spanning tree in the training
split so that the effective-resistance operator is well-defined. The generated
manifest records checksums, split counts, and connectivity checks.
