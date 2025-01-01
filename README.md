# Non-Reciprocal Hopfield Model

Code repository accompanying "Critical Dynamics and Cyclic Memory Retrieval in Non-reciprocal Hopfield Networks" (manuscript submitted for review, 2024).


## Overview

This repository implements both analytical frameworks and numerical methods to study criticality and phase transitions in a non-reciprocal Hopfield model.

### `src/`
Contains core implementations include numerical and various analytical approaches. some examples:
- `glauber.py`: Monte Carlo implementation with Glauber dynamics
- `fold_criticality.py`: Critical characterization along the fold bifurcation line
- `hopf_criticality.py`: Critical characterization along the Hopf bifurcation line
- `liouvillian.py`: Master equation evolution using Liouvillian formalism
- `meanfield.py`: Mean-field phase diagram and bifurcation analysis
- Comparative analysis between Monte Carlo simulations, mean-field theory, and master equation approaches

### `data/`
Contains some simulation outputs and numerical solutions. While not comprehensive, the provided datasets should be sufficient for result verification and further exploration of the phenomena discussed in our paper.

## Citation

Manuscript under review. Citation details will be provided upon publication.

## License

MIT License. See [LICENSE](LICENSE) file.