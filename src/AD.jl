"""
Module AD

Provides GPU-compatible wrappers for automatic differentiation functions of the Enzyme.jl package. Consult the Enzyme documentation to learn how to use the wrapped functions.

# Usage
    import ParallelStencil.AD

# Functions
- `autodiff_deferred!`: wraps function `autodiff_deferred`, promoting all arguments that are not Enzyme.Annotations to Enzyme.Const.
- `autodiff_deferred_thunk!`: wraps function `autodiff_deferred_thunk`, promoting all arguments that are not Enzyme.Annotations to Enzyme.Const.

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
        @parallel configcall=f!(A, B, a) AD.autodiff_deferred!(Enzyme.Reverse, f!, Const, Duplicated(A, Ā), DuplicatedNoNeed(B, B̄), Const(a)) # automatic differentiation of f!
        
        return
    end

    main()

To see a description of a function type `?<functionname>`.
"""
module AD
import ..ParallelKernel.AD: init_AD, autodiff_deferred!, autodiff_deferred_thunk!
export autodiff_deferred!, autodiff_deferred_thunk!
end # Module AD
