# AmplitudeEstimation

Install dependencies and run `python modified_iqae.py`. Edit `config.yaml` to modify experiment parameters. Algorithm plots as shown in the paper are saved in `AmplitudeEstimation/results_images`.


## Config parameters
- **experiments** (`int`): number of experiments to be run.
- **amplitudes** (`List[float]`, `str`): sequence of ground-truth amplitudes $a$ to test. Also accepts code snippets that yield sequences, such as Python's builtin `range` or numpy's `np.linspace`.
- **alpha** (`int`): $\alpha$ as defined in the paper, representing the confidence level or maximum failure rate of the algorithm.
- **epsilons** (`List[float]`, `str`): sequence of maximum error tolerances $\epsilon_i$ to generate a query complexity graph. Accepts code snippets similarly to the **amplitudes** argument.
- **simulator** (`str`): specifies which Qiskit backend to use. Can also pass in the string `'classical'` to simulate circuit measurements using a Bernoulli distribution.
- **plots** (`bool`): writes plots from the paper to `AmplitudeEstimation/results/images` if `True`.
- **verbose** (`bool`): prints intermediate algorithm values if `True`. *Use with caution for high numbers of experiments.*

<!-- 
## Cite this work
If you find our work useful, please consider using the following citation:
```
``` -->
