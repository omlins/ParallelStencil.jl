import ParallelStencil
import Enzyme

function ParallelStencil.ParallelKernel.AD.autodiff_deferred!(args...)
    Enzyme.autodiff_deferred(args...)
    return
end

function ParallelStencil.ParallelKernel.AD.autodiff_deferred_thunk!(args...)
    Enzyme.autodiff_deferred_thunk(args...)
    return
end
