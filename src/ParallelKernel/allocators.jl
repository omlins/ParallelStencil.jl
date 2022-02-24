##
const ZEROS_DOC = """
    @zeros(args...)

!!! note "Advanced"
        @zeros(numbertype, args...)

Call `zeros(numbertype, args...)`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the function `zeros` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (zeros for Threads and CUDA.zeros for CUDA).

!!! note "Advanced"
    The `numbertype` can be explicitly passed as argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the argument `numbertype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.
"""
@doc ZEROS_DOC
macro zeros(args...)
    check_initialized();
    esc(_zeros(args...));
end


##
const ONES_DOC = """
    @ones(args...)

!!! note "Advanced"
        @ones(numbertype, args...)

Call `ones(numbertype, args...)`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the function `ones` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (ones for Threads CUDA.ones for CUDA).

!!! note "Advanced"
    The `numbertype` can be explicitly passed as argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the argument `numbertype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.
"""
@doc ONES_DOC
macro ones(args...)
    check_initialized();
    esc(_ones(args...));
end


##
const RAND_DOC = """
    @rand(args...)

!!! note "Advanced"
        @rand(numbertype, args...)

Call `rand(numbertype, args...)`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the function `rand` is chosen/implemented to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref).

!!! note "Advanced"
    The `numbertype` can be explicitly passed as argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the argument `numbertype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.
"""
@doc RAND_DOC
macro rand(args...)
    check_initialized();
    esc(_rand(args...));
end


## MACROS FORCING PACKAGE, IGNORING INITIALIZATION

macro zeros_cuda(args...)    check_initialized(); esc(_zeros(args...; package=PKG_CUDA)); end
macro ones_cuda(args...)     check_initialized(); esc(_ones(args...; package=PKG_CUDA)); end
macro rand_cuda(args...)     check_initialized(); esc(_rand(args...; package=PKG_CUDA)); end
macro zeros_threads(args...) check_initialized(); esc(_zeros(args...; package=PKG_THREADS)); end
macro ones_threads(args...)  check_initialized(); esc(_ones(args...; package=PKG_THREADS)); end
macro rand_threads(args...)  check_initialized(); esc(_rand(args...; package=PKG_THREADS)); end


## ALLOCATOR FUNCTIONS

function _zeros(args...; package::Symbol=get_package())
    numbertype = get_numbertype()
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.zeros_gpu($numbertype, $(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.zeros_cpu($numbertype, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _ones(args...; package::Symbol=get_package())
    numbertype = get_numbertype()
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.ones_gpu($numbertype, $(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.ones_cpu($numbertype, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _rand(args...; package::Symbol=get_package())
    numbertype = get_numbertype()
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.rand_gpu($numbertype, $(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.rand_cpu($numbertype, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end


## RUNTIME ALLOCATOR FUNCTIONS

zeros_cpu(T0::DataType, T::DataType, args...) = Base.zeros(T, args...)               # If the user has passed a numbertype as first argument to @zeros, then the default numbertype added automatically by the @zeros macro (T0) is not used.
 ones_cpu(T0::DataType, T::DataType, args...) = Base.ones(T, args...)                # ...
 rand_cpu(T0::DataType, T::DataType, args...) = Base.rand(T, args...)                # ...

zeros_gpu(T0::DataType, T::DataType, args...) = CUDA.zeros(T, args...)               # ...
 ones_gpu(T0::DataType, T::DataType, args...) = CUDA.ones(T, args...)                # ...
 rand_gpu(T0::DataType, T::DataType, args...) = CUDA.CuArray(Base.rand(T, args...))  # ...

zeros_cpu(T0::DataType, args...) = (check_default_numbertype(T0); Base.zeros(T0, args...))
 ones_cpu(T0::DataType, args...) = (check_default_numbertype(T0); Base.ones(T0, args...))
 rand_cpu(T0::DataType, args...) = (check_default_numbertype(T0); Base.rand(T0, args...))

zeros_gpu(T0::DataType, args...) = (check_default_numbertype(T0); CUDA.zeros(T0, args...))
 ones_gpu(T0::DataType, args...) = (check_default_numbertype(T0); CUDA.ones(T0, args...))
 rand_gpu(T0::DataType, args...) = (check_default_numbertype(T0); CUDA.CuArray(Base.rand(T0, args...)))

function check_default_numbertype(T0::DataType)
    if (T0 == NUMBERTYPE_NONE) @ArgumentError("the numbertype argument is mandatory in @zeros, @ones and @rand when no default is set.") end
end
