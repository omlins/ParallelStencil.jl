# [50-lines example deployable on GPU and CPU](@id example-single-xpu)

This concise 3-D heat diffusion solver uses ParallelStencil and a simple boolean `USE_GPU` defines whether it runs on GPU or CPU (the number of threads used in the latter case is defined when starting Julia, e.g. with the environment variable [`JULIA_NUM_THREADS`](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_NUM_THREADS)):

```@eval
Main.mdinclude("diffusion3D_novis_noperf.jl")
```

The corresponding file can be found [here](https://github.com/omlins/ParallelStencil.jl/blob/main/examples/diffusion3D_novis_noperf.jl).
