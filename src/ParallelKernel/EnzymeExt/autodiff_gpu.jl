import ParallelStencil
import ParallelStencil: PKG_THREADS
import Enzyme

function ParallelStencil.ParallelKernel.AD.init_AD(package::Symbol)
    if package == PKG_THREADS
        Enzyme.API.runtimeActivity!(true) # NOTE: this is currently required for Enzyme to work correctly with threads
    end
end

function ParallelStencil.ParallelKernel.AD.autodiff_deferred!(arg, args...) # NOTE: minimal specialization is used to avoid overwriting the default method
    Enzyme.autodiff_deferred(arg, args...)
    return
end

function ParallelStencil.ParallelKernel.AD.autodiff_deferred_thunk!(arg, args...) # NOTE: minimal specialization is used to avoid overwriting the default method
    Enzyme.autodiff_deferred_thunk(arg, args...)
    return
end
