import ParallelStencil
import ParallelStencil: PKG_THREADS, PKG_POLYESTER
import Enzyme

# function ParallelStencil.ParallelKernel.AD.init_AD(package::Symbol)
#     if iscpu(package)
#         Enzyme.API.runtimeActivity!(true) # NOTE: this is currently required for Enzyme to work correctly with threads
#     end
# end

# ParallelStencil injects a configuration parameter at the end, for Enzyme we need to wrap that parameter as a Annotation
# for all purposes this ought to be Const. This is not ideal since we might accidentially wrap other parameters the user
# provided as well. This is needed to support @parallel autodiff_deferred(...)
 function promote_to_const(args...)
    ntuple(length(args)) do i
        @inline
        if !(args[i] isa Enzyme.Annotation ||
            (args[i] isa UnionAll && args[i] <: Enzyme.Annotation) || # Const
            (args[i] isa DataType && args[i] <: Enzyme.Annotation)) # Const{Nothing}
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
