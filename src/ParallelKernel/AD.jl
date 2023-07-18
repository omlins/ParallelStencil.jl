"""
Module AD

Provides GPU-compatible wrappers for automatic differentiation functions of the Enzyme.jl package. Consult the Enzyme documentation to learn how to use the wrapped functions.

# Usage
    import ParallelKernel.AD

# Functions
- `autodiff_deferred!`: wraps function `autodiff_deferred`.
- `autodiff_deferred_thunk!`: wraps function `autodiff_deferred_thunk`.

!!! note "Enzyme runtime activity default"
    If ParallelKernel is initialized with Threads, then `Enzyme.API.runtimeActivity!(true)` is called at module load time to ensure correct behavior of Enzyme. If you want to disable this behavior, then call `Enzyme.API.runtimeActivity!(false)` after loading ParallelStencil.

To see a description of a function type `?<functionname>`.
"""
module AD
export autodiff_deferred!, autodiff_deferred_thunk!
import Enzyme

function autodiff_deferred!(args...)
    Enzyme.autodiff_deferred(args...)
    return
end

function autodiff_deferred_thunk!(args...)
    Enzyme.autodiff_deferred_thunk(args...)
    return
end

end # Module AD
