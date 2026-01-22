import ParallelStencil
import ParallelStencil: PKG_THREADS, PKG_POLYESTER
import Enzyme

# NOTE: package specific initialization of Enzyme could be done as follows (not needed in the currently supported versions of Enzyme)
# function ParallelStencil.ParallelKernel.AD.init_AD(package::Symbol)
#     if iscpu(package)
#         Enzyme.API.runtimeActivity!(true) # NOTE: this is currently required for Enzyme to work correctly with threads
#     end
# end

# NOTE: @parallel injects four parameters at the end, which need to be wrapped as Annotations. The current solution is to wrap all
# arguments which are not already Annotations (all the other arguments must be Annotations). Should this change, then one could 
# explicitly wrap just the injected parameters.
function promote_to_const(args::Vararg{Any,N}) where N
    ntuple(Val(N)) do i
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

 function ParallelStencil.ParallelKernel.AD.autodiff_deferred!(mode, f, args::Vararg{Any,N}) where N # NOTE: minimal specialization is required to avoid overwriting the default method
     f    = promote_to_const(f)[1]
     args = promote_to_const(args...)
     Enzyme.autodiff_deferred(mode, f, Enzyme.Const, args...)
     return
 end
 
 function ParallelStencil.ParallelKernel.AD.autodiff_deferred_thunk!(mode, f, args::Vararg{Any,N}) where N # NOTE: minimal specialization is required to avoid overwriting the default method
     f    = promote_to_const(f)[1]
     args = promote_to_const(args...)
     Enzyme.autodiff_deferred_thunk(mode, f, Enzyme.Const, args...)
     return
 end


## FUNCTIONS TO CHECK EXTENSIONS SUPPORT

ParallelStencil.ParallelKernel.is_loaded(::Val{:ParallelStencil_EnzymeExt}) = true
