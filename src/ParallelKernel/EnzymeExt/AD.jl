"""
Module AD

Provides GPU-compatible wrappers for automatic differentiation functions of the Enzyme.jl package. Enzyme needs to be imported before ParallelStencil in order to have it load the corresponding extension. Consult the Enzyme documentation to learn how to use the wrapped functions.

# Usage
    import ParallelKernel.AD

# Functions
- `autodiff_deferred!`: wraps function `Enzyme.autodiff_deferred`, and supports the same arguments, with the exception of the return type activity (3rd argument) which must be omitted and will be automatically inserted (inserting it explicitly will raise a GPU compiler error when targeting a GPU). Additionally, it promotes all arguments that are not `Enzyme.Annotation` to `Enzyme.Const`. As a result, the function can be conveniently called as, e.g., `autodiff_deferred!(Enzyme.Reverse, f!,...)`, instead of the fully explicit form, e.g., `autodiff_deferred!(Enzyme.Reverse, Enzyme.Const(f!), Enzyme.Const,...)` (which is not supported).
- `autodiff_deferred_thunk!`: wraps function `Enzyme.autodiff_deferred_thunk`, in the same way as `autodiff_deferred!` wraps `Enzyme.autodiff_deferred` (see above).

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
