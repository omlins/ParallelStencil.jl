##
const ZEROS_DOC = """
    @zeros(args...)

!!! note "Advanced"
        @zeros(numbertype, args...)

Call `zeros(numbertype, args...)`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the function `zeros` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (zeros for Threads and CUDA.zeros for CUDA).

!!! note "Advanced"
    The `numbertype` can be explicitly passed as argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the argument `numbertype` is mandatory. This needs to be used with care to ensure that no datatype conversions occurs in performance critical computations.
"""
@doc ZEROS_DOC
macro zeros(args...) check_initialized(); esc(_zeros(args...)); end


##
const ONES_DOC = """
    @ones(args...)

!!! note "Advanced"
        @ones(numbertype, args...)

Call `ones(numbertype, args...)`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the function `ones` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (ones for Threads CUDA.ones for CUDA).

!!! note "Advanced"
    The `numbertype` can be explicitly passed as argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the argument `numbertype` is mandatory. This needs to be used with care to ensure that no datatype conversions occurs in performance critical computations.
"""
@doc ONES_DOC
macro ones(args...) check_initialized(); esc(_ones(args...)); end


##
const RAND_DOC = """
    @rand(args...)

!!! note "Advanced"
        @rand(numbertype, args...)

Call `rand(numbertype, args...)`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the function `rand` is chosen/implemented to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref).

!!! note "Advanced"
    The `numbertype` can be explicitly passed as argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the argument `numbertype` is mandatory. This needs to be used with care to ensure that no datatype conversions occurs in performance critical computations.
"""
@doc RAND_DOC
macro rand(args...) check_initialized(); esc(_rand(args...)); end


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
    if !(package in SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    return :(ParallelStencil.ParallelKernel.zeros_xpu($(args...); numbertype_default=$numbertype, package=Symbol($("$package"))))
end

function _ones(args...; package::Symbol=get_package())
    numbertype = get_numbertype()
    if !(package in SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    return :(ParallelStencil.ParallelKernel.ones_xpu($(args...); numbertype_default=$numbertype, package=Symbol($("$package"))))
end

function _rand(args...; package::Symbol=get_package())
    numbertype = get_numbertype()
    if !(package in SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    return :(ParallelStencil.ParallelKernel.rand_xpu($(args...); numbertype_default=$numbertype, package=Symbol($("$package"))))
end

function zeros_xpu(args...; numbertype_default::DataType=NUMBERTYPE_NONE, package::Symbol=PKG_NONE)
    numbertype, args_remainder = determine_args(args...; numbertype_default=numbertype_default, package=package)
    if     (package == PKG_CUDA)    return CUDA.zeros(numbertype, args_remainder...)
    elseif (package == PKG_THREADS) return Base.zeros(numbertype, args_remainder...)
    end
end

function ones_xpu(args...; numbertype_default::DataType=NUMBERTYPE_NONE, package::Symbol=PKG_NONE)
    numbertype, args_remainder = determine_args(args...; numbertype_default=numbertype_default, package=package)
    if     (package == PKG_CUDA)    return CUDA.ones(numbertype, args_remainder...)
    elseif (package == PKG_THREADS) return Base.ones(numbertype, args_remainder...)
    end
end

function rand_xpu(args...; numbertype_default::DataType=NUMBERTYPE_NONE, package::Symbol=PKG_NONE)
    numbertype, args_remainder = determine_args(args...; numbertype_default=numbertype_default, package=package)
    if     (package == PKG_CUDA)    return CUDA.CuArray(rand(numbertype, args_remainder...))
    elseif (package == PKG_THREADS) return Base.rand(numbertype, args_remainder...)
    end
end

function determine_args(args...; numbertype_default::DataType=NUMBERTYPE_NONE, package::Symbol=PKG_NONE)
    if length(args) > 0 && isa(args[1], DataType)
        numbertype     = args[1]
        args_remainder = args[2:end]
    elseif numbertype_default != NUMBERTYPE_NONE
        numbertype = numbertype_default
        args_remainder = args
    else
        @ArgumentError("the numbertype argument is mandatory when no default is set.")
    end
    return numbertype, args_remainder
end
