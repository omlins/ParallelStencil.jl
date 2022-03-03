##
const ZEROS_DOC = """
    @zeros(args...)
    @zeros(args..., <keyword arguments>)

Call `zeros(eltype, args...)`, where `eltype` is by default the `numbertype` selected with [`@init_parallel_kernel`](@ref) and the function `zeros` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (zeros for Threads and CUDA.zeros for CUDA).

!!! note "Advanced"
    The `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Keyword arguments
    - `eltype::DataType`: the type of the elements (numbers).
    - `celldims::Integer|NTuple{N,Integer}=1`: the dimensions of each array cell. Each cell can contain a single value (default) or an N-dimensional array of the specified dimensions.
"""
@doc ZEROS_DOC
macro zeros(args...)
    check_initialized()
    posargs, kwargs_expr = split_args(args)
    eltype, celldims = handle_kwargs_allocators(kwargs_expr, (:eltype, :celldims), "@zeros")
    esc(_zeros(posargs...; eltype=eltype, celldims=celldims))
end


##
const ONES_DOC = """
    @ones(args...)
    @ones(args..., <keyword arguments>)

Call `ones(eltype, args...)`, where `eltype` is by default the `numbertype` selected with [`@init_parallel_kernel`](@ref) and the function `ones` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (ones for Threads and CUDA.ones for CUDA).

!!! note "Advanced"
    The `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Keyword arguments
    - `eltype::DataType`: the type of the elements (numbers).
    - `celldims::Integer|NTuple{N,Integer}=1`: the dimensions of each array cell. Each cell can contain a single value (default) or an N-dimensional array of the specified dimensions.
"""
@doc ONES_DOC
macro ones(args...)
    check_initialized()
    posargs, kwargs_expr = split_args(args)
    eltype, celldims = handle_kwargs_allocators(kwargs_expr, (:eltype, :celldims), "@ones")
    esc(_ones(posargs...; eltype=eltype, celldims=celldims))
end

##
const RAND_DOC = """
    @rand(args...)
    @rand(args...; <keyword arguments>)

Call `rand(eltype, args...)`, where `eltype` is by default the `numbertype` selected with [`@init_parallel_kernel`](@ref) and the function `rand` is chosen/implemented to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref).

!!! note "Advanced"
    The `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Keyword arguments
    - `eltype::DataType`: the type of the elements, which can be numbers, booleans or enums.
    - `celldims::Integer|NTuple{N,Integer}=1`: the dimensions of each array cell. Each cell can contain a single value (default) or an N-dimensional array of the specified dimensions.
"""
@doc RAND_DOC
macro rand(args...)
    check_initialized()
    posargs, kwargs_expr = split_args(args)
    eltype, celldims = handle_kwargs_allocators(kwargs_expr, (:eltype, :celldims), "@rand")
    esc(_rand(posargs...; eltype=eltype, celldims=celldims))
end


##
const FALSES_DOC = """
    @falses(args...)
    @falses(args..., <keyword arguments>)

Call `falses(args...)`, where the function `falses` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref).

# Keyword arguments
    - `celldims::Integer|NTuple{N,Integer}=1`: the dimensions of each array cell. Each cell can contain a single value (default) or an N-dimensional array of the specified dimensions.
"""
@doc FALSES_DOC
macro falses(args...)
    check_initialized()
    posargs, kwargs_expr = split_args(args)
    celldims, = handle_kwargs_allocators(kwargs_expr, (:celldims, :eltype), "@falses")
    esc(_falses(posargs...; celldims=celldims))
end


##
const TRUES_DOC = """
    @trues(args...)
    @trues(args..., <keyword arguments>)

Call `trues(args...)`, where the function `trues` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref).

# Keyword arguments
    - `celldims::Integer|NTuple{N,Integer}=1`: the dimensions of each array cell. Each cell can contain a single value (default) or an N-dimensional array of the specified dimensions.
"""
@doc TRUES_DOC
macro trues(args...)
    check_initialized()
    posargs, kwargs_expr = split_args(args)
    celldims, = handle_kwargs_allocators(kwargs_expr, (:celldims, :eltype), "@trues")
    esc(_trues(posargs...; celldims=celldims))
end


##
const FILL_DOC = """
    @fill(x, args...)
    @fill(x, args...; <keyword arguments>)

Call `fill(convert(eltype, x), args...)`, where `eltype` is by default the `numbertype` selected with [`@init_parallel_kernel`](@ref) and the function `fill` is chosen/implemented to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref).

!!! note "Advanced"
    The element type `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Keyword arguments
    - `eltype::DataType`: the type of the elements, which can be numbers, booleans or enums.
    - `celldims::Integer|NTuple{N,Integer}=1`: the dimensions of each array cell. Each cell can contain a single value (default) or an N-dimensional array of the specified dimensions.
"""
@doc FILL_DOC
macro fill(args...)
    check_initialized()
    posargs, kwargs_expr = split_args(args)
    eltype, celldims = handle_kwargs_allocators(kwargs_expr, (:eltype, :celldims), "@fill")
    esc(_fill(posargs...; eltype=eltype, celldims=celldims))
end


##
const FILL!_DOC = """
    @fill!(A, x)
    @fill!(A, x)

Call `fill!(A, x)`, where the function `fill` is chosen/implemented to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref).

# Arguments
    - `A::Array|ArrayOfArray|TArray|TArrayOfArray`: the array to be filled with `x`.
    - `x::Number|Enum|Collection{Number|Enum, celldims}`: the content to fill `A` with. If `A` is an ArrayOfArray, then `x` can be either a single value or a collection of values (e.g. an array, tuple,...) of the size `celldims` of `A`.
"""
@doc FILL!_DOC
macro fill!(args...) check_initialized(); esc(_fill!(args...)); end


## MACROS FORCING PACKAGE, IGNORING INITIALIZATION

macro zeros_cuda(args...)     check_initialized(); esc(_zeros(args...; package=PKG_CUDA)); end
macro ones_cuda(args...)      check_initialized(); esc(_ones(args...; package=PKG_CUDA)); end
macro rand_cuda(args...)      check_initialized(); esc(_rand(args...; package=PKG_CUDA)); end
macro falses_cuda(args...)    check_initialized(); esc(_falses(args...; package=PKG_CUDA)); end
macro trues_cuda(args...)     check_initialized(); esc(_trues(args...; package=PKG_CUDA)); end
macro fill_cuda(args...)      check_initialized(); esc(_fill(args...; package=PKG_CUDA)); end
macro fill!_cuda(args...)     check_initialized(); esc(_fill!(args...; package=PKG_CUDA)); end
macro zeros_threads(args...)  check_initialized(); esc(_zeros(args...; package=PKG_THREADS)); end
macro ones_threads(args...)   check_initialized(); esc(_ones(args...; package=PKG_THREADS)); end
macro rand_threads(args...)   check_initialized(); esc(_rand(args...; package=PKG_THREADS)); end
macro falses_threads(args...) check_initialized(); esc(_falses(args...; package=PKG_THREADS)); end
macro trues_threads(args...)  check_initialized(); esc(_trues(args...; package=PKG_THREADS)); end
macro fill_threads(args...)   check_initialized(); esc(_fill(args...; package=PKG_THREADS)); end
macro fill!_threads(args...)  check_initialized(); esc(_fill!(args...; package=PKG_THREADS)); end


## ALLOCATOR FUNCTIONS

function _zeros(args...; eltype=nothing, celldims=nothing, package::Symbol=get_package())
    celltype = determine_celltype(eltype, celldims)
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.zeros_gpu($celltype, $(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.zeros_cpu($celltype, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _ones(args...; eltype=nothing, celldims=nothing, package::Symbol=get_package())
    celltype = determine_celltype(eltype, celldims)
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.ones_gpu($celltype, $(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.ones_cpu($celltype, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _rand(args...; eltype=nothing, celldims=nothing, package::Symbol=get_package())
    celltype = determine_celltype(eltype, celldims)
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.rand_gpu($celltype, $(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.rand_cpu($celltype, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _falses(args...; celldims=nothing, package::Symbol=get_package())
    celltype = determine_celltype(Bool, celldims)
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.falses_gpu($celltype, $(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.falses_cpu($celltype, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _trues(args...; celldims=nothing, package::Symbol=get_package())
    celltype = determine_celltype(Bool, celldims)
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.trues_gpu($celltype, $(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.trues_cpu($celltype, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _fill(args...; eltype=nothing, celldims=nothing, package::Symbol=get_package())
    celltype = determine_celltype(eltype, celldims)
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.fill_gpu($celltype, $(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.fill_cpu($celltype, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _fill!(args...; package::Symbol=get_package())
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.fill_gpu!($(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.fill_cpu!($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function determine_celltype(eltype, celldims)
    if isnothing(eltype)
        eltype = get_numbertype()
        if (eltype == NUMBERTYPE_NONE) @ArgumentError("the keyword argument 'eltype' is mandatory in @zeros, @ones, @rand and @fill when no default is set.") end
    end
    if !isnothing(celldims) celltype = :(ParallelStencil.ParallelKernel.SArray{Tuple{$celldims...}, $eltype, length($celldims), prod($celldims)})
    else                    celltype = eltype
    end
    return celltype
end


## MACRO ARGUMENT HANDLER FUNCTIONS

function handle_kwargs_allocators(kwargs_expr, valid_kwargs, macroname)
    kwargs = split_kwargs(kwargs_expr)
    validate_kwargkeys(kwargs, valid_kwargs, macroname)
    return extract_kwargvalues(kwargs, valid_kwargs)
end


## RUNTIME ALLOCATOR FUNCTIONS

 zeros_cpu(T::DataType, args...) = (check_datatype(T); Base.zeros(T, args...))
  ones_cpu(T::DataType, args...) = (check_datatype(T); fill_cpu(T, 1, args...))
  rand_cpu(T::DataType, args...) = (check_datatype(T, Bool, Enum); Base.rand(T, args...))
falses_cpu(T::DataType, args...) = Base.zeros(T, args...)
 trues_cpu(T::DataType, args...) = fill_cpu(T, true, args...)

 zeros_gpu(T::DataType, args...) = (check_datatype(T); (T <: Number) ? CUDA.zeros(T, args...) : CuArray(zeros_cpu(T, args...)))
  ones_gpu(T::DataType, args...) = (check_datatype(T); (T <: Number) ? CUDA.ones(T, args...) : CuArray(ones_cpu(T, args...)))
  rand_gpu(T::DataType, args...) = CuArray(rand_cpu(T, args...))
falses_gpu(T::DataType, args...) = CuArray(falses_cpu(T, args...))
 trues_gpu(T::DataType, args...) = CuArray(trues_cpu(T, args...))
  fill_gpu(T::DataType, args...) = CuArray(fill_cpu(T, args...))

function fill_cpu(T::DataType, x, args...)::Array{T}
    if (!(eltype(x) <: Number) || (eltype(x) == Bool)) && (eltype(x) != eltype(T)) @ArgumentError("@fill: the (element) type of argument 'x' is not a normal number type ($(eltype(x))), but does not match the obtained (default) 'eltype' ($(eltype(T))); automatic conversion to $(eltype(T)) is therefore not attempted. Set the keyword argument 'eltype' accordingly to the element type of 'x' or pass an 'x' of a different (element) type.") end
    check_datatype(T, Bool, Enum)
    if T <: SArray
        if     (length(x) == length(T)) cell = convert(T, x)
        elseif (length(x) == 1)         cell = convert(T, fill(convert(eltype(T), x), size(T)))
        else                            @ArgumentError("@fill: argument 'x' contains the wrong number of elements ($(length(x))). It must be a scalar or contain the number of elements defined by 'celldims'.")
        end
    else
        cell = convert(T, x)
    end
    Base.fill(cell, args...)
end

fill_cpu!(A, x) = Base.fill!(A, construct_cell(A, x))
fill_gpu!(A, x) = CUDA.fill!(A, construct_cell(A, x))

function construct_cell(A, x)
    T_cell = eltype(A)
    if (!(eltype(x) <: Number) || (eltype(x) == Bool)) && (eltype(x) != eltype(T_cell)) @ArgumentError("@fill!: the (element) type of argument 'x' is not a normal number type ($(eltype(x))), but does not match the type of the elements of 'A' ($(eltype(T_cell))); automatic conversion to $(eltype(T_cell)) is therefore not attempted. Pass an 'x' of a different (element) type.") end
    if T_cell <: SArray
        if     (length(x) == length(T_cell)) cell = convert(T_cell, x)
        elseif (length(x) == 1)              cell = convert(T_cell, fill(convert(eltype(T_cell), x), size(T_cell)))
        else                                 @ArgumentError("@fill!: argument 'x' contains the wrong number of elements ($(length(x))). It must be a scalar or contain the number of elements defined by 'celldims'.")
        end
    else
        cell = convert(T_cell, x)
    end
    return cell
end

function check_datatype(T::DataType, valid_non_numbertypes::Union{DataType, UnionAll}...)
    if !any(T .<: valid_non_numbertypes) && !any(eltype(T) .<: valid_non_numbertypes)
        check_numbertype(eltype(T))
    end
end

import Base.length
length(x::Enum) = 1

import Base.eltype
eltype(x::Enum) = typeof(x)

import Random
Random.SamplerType{T}() where {T<:Enum} = Random.Sampler(Random.GLOBAL_RNG, instances(T))
