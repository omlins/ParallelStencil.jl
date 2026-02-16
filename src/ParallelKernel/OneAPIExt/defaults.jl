const ERRMSG_ONEAPIEXT_NOT_LOADED = "the oneAPI extension was not loaded. Make sure to import oneAPI before ParallelStencil."


# select_hardware.jl

handle_kernelabstractions_oneapi(arg...)  = @NotLoadedError(ERRMSG_ONEAPIEXT_NOT_LOADED)
