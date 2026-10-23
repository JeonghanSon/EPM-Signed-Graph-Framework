# Datasets

The experiments use public signed networks from the Stanford Network Analysis
Project (SNAP). Use the source pages below and cite the corresponding dataset
publications when reporting results.

| Dataset key | SNAP source page | Expected local filename |
|---|---|---|
| `bitcoinalpha` | [Bitcoin Alpha](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html) | `soc-sign-bitcoinalpha.csv` |
| `bitcoinotc` | [Bitcoin OTC](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html) | `soc-sign-bitcoinotc.csv` |
| `wiki_elec` | [Wiki-Elec](https://snap.stanford.edu/data/wiki-Elec.html) | `wikiElec.ElecBs3.txt` |
| `wiki_rfa` | [Wiki-RfA](https://snap.stanford.edu/data/wiki-RfA.html) | `wiki-RfA.txt` |
| `slashdot` | [Slashdot](https://snap.stanford.edu/data/soc-sign-Slashdot090221.html) | `soc-sign-Slashdot090221.txt` |
| `epinions` | [Epinions](https://snap.stanford.edu/data/soc-sign-epinions.html) | `soc-sign-epinions.txt` |

For a lightweight default run, this repository includes
`data/raw/soc-sign-bitcoinalpha.csv`. Download the other datasets from SNAP and
place them under `data/raw/` using the names listed in
`data/metadata/raw_files.json`.

The repository's MIT License applies to the implementation, not to third-party
dataset files. The included BTC-Alpha copy and any additionally downloaded
datasets retain their original source terms and attribution requirements.

The generated networks used for the controlled synthetic validation are
included under [`data/synthetic/`](synthetic/README.md), together with their
construction metadata and graph fingerprints.

Preprocessing preserves the transductive node set, restricts the experiment to
the largest connected component, and reserves a spanning tree in the training
split so that the effective-resistance operator is well-defined. The generated
manifest records checksums, split counts, and connectivity checks. The paper
and provided experiment scripts use split seed 0.
