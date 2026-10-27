# Testing

The test suite is run with the [Julia package manager](https://docs.julialang.org/en/v1/stdlib/Pkg/):
```julia
import Pkg; Pkg.test("ParallelStencil")
```
Each test file runs in its own Julia process. Test files can be selected by file name (selecting all files of this name, e.g. `test_parallel.jl` exists both in `test/` and in `test/ParallelKernel/`) or by path relative to the `test` folder; `--stop-on-fail` stops the test run at the first test file with failing tests:
```julia
import Pkg; Pkg.test("ParallelStencil"; test_args=["ParallelKernel/test_parallel.jl", "--stop-on-fail"])
```

The GPU packages (CUDA, AMDGPU and Metal) and Enzyme are not dependencies of the test environment (`test/Project.toml`): resolving the GPU packages all together holds each of them back to an old version, and Enzyme typically lags behind new Julia releases. The test runner installs them into the (temporary) test environment of `Pkg.test` as selected with the following environment variables (the tests of a selected GPU backend must run):

| Variable          | Description                                                                                                                               |
|:----------------- |:----------------------------------------------------------------------------------------------------------------------------------------- |
| `PS_TEST_AD`      | Set to `0` to not install Enzyme into the test environment and skip the tests of automatic differentiation (default: `1`). |
| `PS_TEST_BACKEND` | GPU backend to test (`CUDA`, `AMDGPU` or `Metal`): its package is installed into the test environment if it is not already, and the test run fails instead of skipping the tests of this backend if it is not functional. |

Without it, the CPU backends (Threads, Polyester and KernelAbstractions on the CPU) are tested, and the tests of a GPU backend run only if its package is already installed in the test environment and functional. For example, to test a GPU backend:

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

The test environment of `Pkg.test` is temporary, so the installation is repeated at each test run (it is fast once the package is in the depot). To avoid it, the package can be added permanently to the test environment with `julia --project=test -e 'import Pkg; Pkg.add("CUDA")'`; this modifies `test/Project.toml` (do not commit this change) and `test/Manifest.toml` (not tracked by git).

## Continuous integration

- **GitHub Actions** (CPUs, and Apple GPUs on the macOS runners): Julia `min` and `1` on Linux and macOS (Apple silicon), and Julia `pre` on Linux (allowed to fail) (`.github/workflows/ci.yml`).
- **CSCS CI** on [Alps](https://docs.cscs.ch/alps/), with one pipeline per vCluster:
  - [Daint](https://docs.cscs.ch/clusters/daint/) (Nvidia GH200 GPUs): the full test suite on one GH200 node, with the CPU backends running on the Grace CPU and the CUDA and KernelAbstractions backends on the GPU (`ci/cscs-gh200.yml`; [pipelines](https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/7487078532071147/3127735853767612/-/pipelines)).
  - [Beverin](https://docs.cscs.ch/clusters/beverin/) (AMD MI250X and MI300A GPUs): the full test suite on one node per GPU type, with the CPU backends running on the EPYC CPU and the AMDGPU and KernelAbstractions backends on the GPU (`ci/cscs-mi250.yml`, `ci/cscs-mi300.yml`; [MI250X pipelines](https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/7487078532071147/3127735853767613/-/pipelines), [MI300A pipelines](https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/7487078532071147/3127735853767614/-/pipelines)).

  For pull requests by other contributors, a maintainer starts the pipelines by commenting `cscs-ci run`.

  The README badges of the CSCS CI are generated from its commit statuses by the workflow `.github/workflows/CSCSBadges.yml` (branch `badges`).

  Each job sets `PS_TEST_BACKEND` to the GPU backend it tests (`CUDA` on Daint, `AMDGPU` on Beverin and `Metal` on the macOS runners of GitHub Actions); the Linux jobs of GitHub Actions test the CPU backends only. The newest KernelAbstractions (0.10) and AMDGPU (2.9) require LLVM.jl 10, which Enzyme does not support yet: one Linux job and the MI300 job run without Enzyme (`PS_TEST_AD=0`) to test them, the other jobs test with Enzyme (KernelAbstractions 0.9).

Coverage of all jobs is uploaded to [Codecov](https://app.codecov.io/gh/omlins/ParallelStencil.jl) (flags `cpu`, `gh200`, `mi250` and `mi300`).
