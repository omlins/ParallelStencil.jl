"""
Module AD

Provides GPU-compatible wrappers for automatic differentiation functions of the Enzyme.jl package. Enzyme needs to be imported before ParallelStencil in order to have it load the corresponding extension. Consult the Enzyme documentation to learn how to use the wrapped functions.

# Usage
    import ParallelKernel.AD

# Functions
- `autodiff_deferred!(mode, f, args...)`: wraps function `Enzyme.autodiff_deferred`, calling it as `Enzyme.autodiff_deferred(mode, f, Const, args...)`, promoting all arguments of `args` and `f` that are not `Enzyme.Annotation` to Enzyme.Const (the return type activity is always `Enzyme.Const`, as `@parallel` and `@parallel_indices` functions must return nothing).
- `autodiff_deferred_thunk!`: wraps function `autodiff_deferred_thunk`, promoting all arguments that are not Enzyme.Annotations to Enzyme.Const. The function signature is `autodiff_deferred_thunk!(mode, f, args...)` and the return type must not be fixed to `Const` as @parallel and @parallel_indices functions must return `nothing`.

To see a description of a function type `?<functionname>`.
"""
module AD
using ..Exceptions

const ERRMSG_ENZYMEEXT_NOT_LOADED = "AD: the Enzyme extension was not loaded. Make sure to import Enzyme before ParallelStencil."

init_AD(args...)                  = return                                            # NOTE: a call will be triggered from @init_parallel_kernel, but it will do nothing if the extension is not loaded. Methods are to be defined in the AD extension modules.
autodiff_deferred!(args...)       = @NotLoadedError(ERRMSG_ENZYMEEXT_NOT_LOADED)
autodiff_deferred_thunk!(args...) = @NotLoadedError(ERRMSG_ENZYMEEXT_NOT_LOADED)

export autodiff_deferred!, autodiff_deferred_thunk!

end # Module AD
