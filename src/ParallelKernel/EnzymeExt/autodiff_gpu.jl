import ParallelStencil
import ParallelStencil: PKG_THREADS, PKG_POLYESTER
import Enzyme

# function ParallelStencil.ParallelKernel.AD.init_AD(package::Symbol)
#     if iscpu(package)
#         Enzyme.API.runtimeActivity!(true) # NOTE: this is currently required for Enzyme to work correctly with threads
#     end
# end

function promoto_to_const(args...)
    ntuple(length(args)) do i
        @inbounds
        if !(args[i] isa Enzyme.Annotation)
            return Enzyme.Const(args[i])
        else
            return args[i]
        end
    end
end

function ParallelStencil.ParallelKernel.AD.autodiff_deferred!(arg, args...) # NOTE: minimal specialization is used to avoid overwriting the default method
    args = promote_to_const(args...)
    Enzyme.autodiff_deferred(arg, args...)
    return
end

function ParallelStencil.ParallelKernel.AD.autodiff_deferred_thunk!(arg, args...) # NOTE: minimal specialization is used to avoid overwriting the default method
    args = promote_to_const(args...)
    Enzyme.autodiff_deferred_thunk(arg, args...)
    return
end


## FUNCTIONS TO CHECK EXTENSIONS SUPPORT

ParallelStencil.ParallelKernel.is_loaded(::Val{:ParallelStencil_EnzymeExt}) = true
