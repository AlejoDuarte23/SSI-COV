
# SSICOV Class

The `SSICOV` class a system identification method for structural health monitoring (SHM) using Stochastic Subspace Identification (SSI). It processes acceleration data to extract modal parameters of a structure.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Initialization

Create an instance of the `SSICOV` class with the required parameters.

```python
import numpy as np
from SSICOV import SSICOV

ssicov = SSICOV(acc, fs, Ts, Nc, Nmax, Nmin)
```

### Methods

#### `NexT`

Calculates the Impulse Response Function (IRF) using cross-correlation of the input acceleration data.

```python
IRF = ssicov.NexT()
```

#### `blockToeplitz`

Generates the block Toeplitz matrix and performs Singular Value Decomposition (SVD).

```python
U, S, V, T1 = ssicov.blockToeplitz(IRF)
```

#### `modalID`

Identifies modal parameters from the SVD components.

```python
fn, zeta, phi = ssicov.modalID(U, S, Nmodes, Nyy, fs)
```

#### `stabilityCheck`

Checks the stability of the identified modes.

```python
fn, zeta, phi, MAC, stability_status = ssicov.stabilityCheck(fn0, zeta0, phi0, fn1, zeta1, phi1)
```

#### `getStablePoles`

Filters and returns the stable poles from the identified modes.

```python
fnS, zetaS, phiS, MACS = ssicov.getStablePoles(fn, zeta, phi, MAC, stability_status)
```

#### `run`

Runs the complete SSI process and returns the stable modal parameters.

```python
fnS, zetaS, phiS, MACS, stability_status, fn2 = ssicov.run()
```

## Utilities

The class uses several utility functions for timing and size printing. Ensure to include these utility functions in your project.

```python
from utils import print_input_sizes, timeit
```
