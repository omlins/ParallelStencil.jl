"""
Module AD

Provides GPU-compatible functions for automatic differentiation. The functions rely on the Enzyme.jl package.

# Usage
    using ParallelStencil.AD

# Functions
- `autodiff_deferred!`
- `autodiff_deferred_thunk!`

# Examples
    const USE_GPU = true
    using ParallelStencil
    using ParallelStencil.AD, Enzyme
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
        @parallel configcall=f!(A, B, a) autodiff_deferred!(Enzyme.Reverse, f!, Duplicated(A, Ā), Duplicated(B, B̄), Const(a)) # automatic differentiation of f!
        
        return
    end

    main()

To see a description of a function type `?<functionname>`.
"""
module AD
export autodiff_deferred!, autodiff_deferred_thunk!
import ..ParallelKernel.AD: autodiff_deferred!, autodiff_deferred_thunk!

end # Module AD
