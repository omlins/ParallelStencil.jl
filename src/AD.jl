"""
Module AD

Provides GPU-compatible wrappers for automatic differentiation functions of the Enzyme.jl package. Consult the Enzyme documentation to learn how to use the wrapped functions.

# Usage
    import ParallelStencil.AD

# Functions
- `autodiff_deferred!`: wraps function `Enzyme.autodiff_deferred`, and supports the same arguments, with the exception of the return type activity (3rd argument) which must be omitted and will be automatically inserted (inserting it explicitly will raise a GPU compiler error when targeting a GPU). Additionally, it promotes all arguments that are not `Enzyme.Annotation` to `Enzyme.Const`. As a result, the function can be conveniently called as, e.g., `autodiff_deferred!(Enzyme.Reverse, f!,...)`, instead of the fully explicit form, e.g., `autodiff_deferred!(Enzyme.Reverse, Enzyme.Const(f!), Enzyme.Const,...)` (which is not supported).
- `autodiff_deferred_thunk!`: wraps function `Enzyme.autodiff_deferred_thunk`, in the same way as `autodiff_deferred!` wraps `Enzyme.autodiff_deferred` (see above).


# Examples
    const USE_GPU = true
    using ParallelStencil
    import ParallelStencil.AD
    using Enzyme
    @static if USE_GPU
        @init_parallel_stencil(CUDA, Float64, 3);
    else
        @init_parallel_stencil(Threads, Float64, 3);
    end

    @parallel_indices (ix) function f!(A, B, a)
        A[ix] += a * B[ix] * 100.65
        return
    end

    function main()
        N = 16
        a = 6.5
        A = @rand(N)
        B = @rand(N)
        Ā = @ones(size(A))
        B̄ = @ones(size(B))

        @info "running on CPU/GPU"
        @parallel f!(A, B, a) # normal call of f!
        @parallel configcall=f!(A, B, a) AD.autodiff_deferred!(Enzyme.Reverse, f!, Duplicated(A, Ā), DuplicatedNoNeed(B, B̄), a) # automatic differentiation of f!
        
        return
    end

    main()

To see a description of a function type `?<functionname>`.
"""
module AD
import ..ParallelKernel.AD: init_AD, autodiff_deferred!, autodiff_deferred_thunk!
export autodiff_deferred!, autodiff_deferred_thunk!
end # Module AD
