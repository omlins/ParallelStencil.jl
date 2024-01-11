"""
Module AD

Provides GPU-compatible wrappers for automatic differentiation functions of the Enzyme.jl package. Consult the Enzyme documentation to learn how to use the wrapped functions.

# Usage
    import ParallelKernel.AD

# Functions
- `autodiff_deferred!`: wraps function `autodiff_deferred`.
- `autodiff_deferred_thunk!`: wraps function `autodiff_deferred_thunk`.

!!! note "Enzyme runtime activity default"
    If ParallelKernel is initialized with Threads, then `Enzyme.API.runtimeActivity!(true)` is called to ensure correct behavior of Enzyme. If you want to disable this behavior, then call `Enzyme.API.runtimeActivity!(false)` after loading ParallelStencil.

To see a description of a function type `?<functionname>`.
"""
module AD
export autodiff_deferred!, autodiff_deferred_thunk!

function autodiff_deferred! end
function autodiff_deferred_thunk! end

end # Module AD
