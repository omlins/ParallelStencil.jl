# Testing

The test suite is run with the [Julia package manager](https://docs.julialang.org/en/v1/stdlib/Pkg/):
```julia
import Pkg; Pkg.test("ParallelStencil")
```
Each test file runs in its own Julia process. Test files can be selected by file name (selecting all files of this name, e.g. `test_parallel.jl` exists both in `test/` and in `test/ParallelKernel/`) or by path relative to the `test` folder; `--stop-on-fail` stops the test run at the first test file with failing tests:
```julia
import Pkg; Pkg.test("ParallelStencil"; test_args=["ParallelKernel/test_parallel.jl", "--stop-on-fail"])
```

The tests of the CUDA, AMDGPU and Metal backends run if the corresponding package is functional, and are skipped otherwise. The following environment variable controls the test run:

| Variable          | Description                                                                                                                               |
|:----------------- |:----------------------------------------------------------------------------------------------------------------------------------------- |
| `PS_TEST_BACKEND` | GPU backend that must be functional (`CUDA`, `AMDGPU` or `Metal`); if it is not, the test run fails instead of skipping the tests of this backend. |

For example, to make sure that the GPU tests are run:

:::code-group

```bash [Nvidia GPUs]
export PS_TEST_BACKEND=CUDA
julia --project -e 'import Pkg; Pkg.test()'
```

```bash [AMD GPUs]
export PS_TEST_BACKEND=AMDGPU
julia --project -e 'import Pkg; Pkg.test()'
```

```bash [Apple GPUs]
export PS_TEST_BACKEND=Metal
julia --project -e 'import Pkg; Pkg.test()'
```

:::

## Continuous integration

- **GitHub Actions** (CPUs, and Apple GPUs on the macOS runners): Julia `min` and `1` on Linux and macOS (Apple silicon), and Julia `pre` on Linux (allowed to fail) (`.github/workflows/ci.yml`).
- **CSCS CI** on [Alps](https://docs.cscs.ch/alps/), with one pipeline per vCluster:
  - [Daint](https://docs.cscs.ch/clusters/daint/) (Nvidia GH200 GPUs): the full test suite on one GH200 node, with the CPU backends running on the Grace CPU and the CUDA and KernelAbstractions backends on the GPU (`ci/cscs-gh200.yml`; [pipelines](https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/7487078532071147/3127735853767612/-/pipelines)).
  - [Beverin](https://docs.cscs.ch/clusters/beverin/) (AMD MI250X and MI300A GPUs): the full test suite on one node per GPU type, with the CPU backends running on the EPYC CPU and the AMDGPU and KernelAbstractions backends on the GPU (`ci/cscs-mi250.yml`, `ci/cscs-mi300.yml`; [MI250X pipelines](https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/7487078532071147/3127735853767613/-/pipelines), [MI300A pipelines](https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/7487078532071147/3127735853767614/-/pipelines)).

  For pull requests by other contributors, a maintainer starts the pipelines by commenting `cscs-ci run`.

Coverage of all jobs is uploaded to [Codecov](https://app.codecov.io/gh/omlins/ParallelStencil.jl) (flags `cpu`, `gh200`, `mi250` and `mi300`).
