# Testing

The test suite is run with the [Julia package manager](https://docs.julialang.org/en/v1/stdlib/Pkg/):
```julia
import Pkg; Pkg.test("ParallelStencil")
```
Each test file runs in its own Julia process. Test files can be selected by name, and `--stop-on-fail` stops the test run at the first test file with failing tests:
```julia
import Pkg; Pkg.test("ParallelStencil"; test_args=["test_parallel.jl", "--stop-on-fail"])
```

The tests of the CUDA, AMDGPU and Metal backends run if the corresponding package is functional, and are skipped otherwise.

## Continuous integration

- **GitHub Actions** (CPU): Julia `1.10`, `1.11` and `1` on Linux and macOS (`.github/workflows/ci.yml`).
