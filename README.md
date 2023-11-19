# gdf: Gaussian density fitting for periodic solids

The `gdf` package implements various Gaussian density-fitting (GDF) models for periodic solids in quantum chemistry.

### Installation

From source:

```bash
git clone https://github.com/obackhouse/gdf
pip install .
```

### Usage

The implemented models are built upon the [`pyscf`](https://github.com/pyscf/pyscf) functionality:

```python
from pyscf.pbc import gto, scf
from gdf import CCGDF
cell = gto.Cell()
cell.atom = "He 0 0 0; He 1 1 1"
cell.a = np.eye(3) * 3
cell.basis = "6-31g"
cell.build()
kpts = cell.make_kpts([3, 3, 3])
df = CCGDF(cell, kpts)
df.build()
```
