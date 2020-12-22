##
const ZEROS_DOC = """
    @zeros(args...)

Call `zeros(numbertype, args...)`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the function `zeros` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (zeros for Threads and CUDA.zeros for CUDA).
"""
@doc ZEROS_DOC
macro zeros(args...) check_initialized(); esc(_zeros(args...)); end


##
const ONES_DOC = """
    @ones(args...)

Call `ones(numbertype, args...)`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the function `ones` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (ones for Threads CUDA.ones for CUDA).
"""
@doc ONES_DOC
macro ones(args...) check_initialized(); esc(_ones(args...)); end


##
const RAND_DOC = """
    @rand(args...)

Call `rand(numbertype, args...)`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the function `rand` is chosen/implemented to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref).
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
    if     (package == PKG_CUDA)    return :(CUDA.zeros($numbertype, $(args...)))
    elseif (package == PKG_THREADS) return :(Base.zeros($numbertype, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _ones(args...; package::Symbol=get_package())
    numbertype = get_numbertype()
    if     (package == PKG_CUDA)    return :(CUDA.ones($numbertype, $(args...)))
    elseif (package == PKG_THREADS) return :(Base.ones($numbertype, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _rand(args...; package::Symbol=get_package())
    numbertype = get_numbertype()
    if     (package == PKG_CUDA)    return :(CUDA.CuArray(rand($numbertype, $(args...))))
    elseif (package == PKG_THREADS) return :(Base.rand($numbertype, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end
