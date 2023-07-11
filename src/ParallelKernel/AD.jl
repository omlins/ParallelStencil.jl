"""
Module AD

Provides GPU-compatible functions for automatic differentiation. The functions rely on the Enzyme.jl package.

# Usage
    using ParallelKernel.AD

# Functions
- `autodiff_deferred!`
- `autodiff_deferred_thunk!`

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
