##
const ZEROS_DOC = """
    @zeros(args...)
    @zeros(args..., <keyword arguments>)

Call `zeros(eltype, args...)`, where `eltype` is by default the `numbertype` selected with [`@init_parallel_kernel`](@ref) and the function `zeros` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (zeros for Threads, CUDA.zeros for CUDA and AMDGPU.zeros for AMDGPU).

!!! note "Advanced"
    The `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Keyword arguments
- `eltype::DataType`: the type of the elements (numbers).
- `celldims::Integer|NTuple{N,Integer}=1`: the dimensions of each array cell. Each cell can contain a single value (default) or an N-dimensional array of the specified dimensions.
!!! note "Advanced"
    - `celltype::DataType`: the type of each array cell; it must be generated with the macro `@CellType`. The keyword argument `celltype` is incompatible with the other keyword arguments: if any of them is set, then the `celltype` is automatically defined. The `celltype` needs only to be specified to use named cell fields. Note that values can always be addressed with array indices, even when cell field names are defined.
    - `blocklength::Integer`: refers to the amount of values of a same `Cell` field that are stored contigously (`blocklength=1` means array of struct like storage; `blocklength=prod(dims)` means array struct of array like storage; `blocklength=0` is an alias for `blocklength=prod(dims)`, enabling better peformance thanks to more specialized dispatch). By default, `blocklength` is automatically set to `0` if a GPU package was chosen with [`@init_parallel_kernel`](@ref) and to `1` if a CPU package was chosen. Furthermore, the argument `blocklength` is only of effect if either `celldims` or `celltype` is set, else it is ignored.

See also: [`@ones`](@ref), [`@rand`](@ref), [`@falses`](@ref), [`@trues`](@ref), [`@fill`](@ref), [`@CellType`](@ref)
"""
@doc ZEROS_DOC
macro zeros(args...)
    check_initialized()
    posargs, kwargs_expr = split_args(args)
    eltype, celldims, celltype, blocklength = extract_kwargvalues(kwargs_expr, (:eltype, :celldims, :celltype, :blocklength), "@zeros")
    esc(_zeros(posargs...; eltype=eltype, celldims=celldims, celltype=celltype, blocklength=blocklength))
end


##
const ONES_DOC = """
    @ones(args...)
    @ones(args..., <keyword arguments>)

Call `ones(eltype, args...)`, where `eltype` is by default the `numbertype` selected with [`@init_parallel_kernel`](@ref) and the function `ones` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (ones for Threads, CUDA.ones for CUDA and AMDGPU.ones for AMDGPU).

!!! note "Advanced"
    The `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Keyword arguments
- `eltype::DataType`: the type of the elements (numbers).
- `celldims::Integer|NTuple{N,Integer}=1`: the dimensions of each array cell. Each cell can contain a single value (default) or an N-dimensional array of the specified dimensions.
!!! note "Advanced"
    - `celltype::DataType`: the type of each array cell; it must be generated with the macro `@CellType`. The keyword argument `celltype` is incompatible with the other keyword arguments: if any of them is set, then the `celltype` is automatically defined. The `celltype` needs only to be specified to use named cell fields. Note that values can always be addressed with array indices, even when cell field names are defined.
    - `blocklength::Integer`: refers to the amount of values of a same `Cell` field that are stored contigously (`blocklength=1` means array of struct like storage; `blocklength=prod(dims)` means array struct of array like storage; `blocklength=0` is an alias for `blocklength=prod(dims)`, enabling better peformance thanks to more specialized dispatch). By default, `blocklength` is automatically set to `0` if a GPU package was chosen with [`@init_parallel_kernel`](@ref) and to `1` if a CPU package was chosen. Furthermore, the argument `blocklength` is only of effect if either `celldims` or `celltype` is set, else it is ignored.

See also: [`@zeros`](@ref), [`@rand`](@ref), [`@falses`](@ref), [`@trues`](@ref), [`@fill`](@ref), [`@CellType`](@ref)
"""
@doc ONES_DOC
macro ones(args...)
    check_initialized()
    posargs, kwargs_expr = split_args(args)
    eltype, celldims, celltype, blocklength = extract_kwargvalues(kwargs_expr, (:eltype, :celldims, :celltype, :blocklength), "@ones")
    esc(_ones(posargs...; eltype=eltype, celldims=celldims, celltype=celltype, blocklength=blocklength))
end

##
const RAND_DOC = """
    @rand(args...)
    @rand(args..., <keyword arguments>)

Call `rand(eltype, args...)`, where `eltype` is by default the `numbertype` selected with [`@init_parallel_kernel`](@ref) and the function `rand` is chosen/implemented to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref).

!!! note "Advanced"
    The `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Keyword arguments
- `eltype::DataType`: the type of the elements, which can be numbers, booleans or enums.
- `celldims::Integer|NTuple{N,Integer}=1`: the dimensions of each array cell. Each cell can contain a single value (default) or an N-dimensional array of the specified dimensions.
!!! note "Advanced"
    - `celltype::DataType`: the type of each array cell; it must be generated with the macro `@CellType`. The keyword argument `celltype` is incompatible with the other keyword arguments: if any of them is set, then the `celltype` is automatically defined. The `celltype` needs only to be specified to use named cell fields. Note that values can always be addressed with array indices, even when cell field names are defined.
    - `blocklength::Integer`: refers to the amount of values of a same `Cell` field that are stored contigously (`blocklength=1` means array of struct like storage; `blocklength=prod(dims)` means array struct of array like storage; `blocklength=0` is an alias for `blocklength=prod(dims)`, enabling better peformance thanks to more specialized dispatch). By default, `blocklength` is automatically set to `0` if a GPU package was chosen with [`@init_parallel_kernel`](@ref) and to `1` if a CPU package was chosen. Furthermore, the argument `blocklength` is only of effect if either `celldims` or `celltype` is set, else it is ignored.

See also: [`@zeros`](@ref), [`@ones`](@ref), [`@falses`](@ref), [`@trues`](@ref), [`@fill`](@ref), [`@CellType`](@ref)
"""
@doc RAND_DOC
macro rand(args...)
    check_initialized()
    posargs, kwargs_expr = split_args(args)
    eltype, celldims, celltype, blocklength = extract_kwargvalues(kwargs_expr, (:eltype, :celldims, :celltype, :blocklength), "@rand")
    esc(_rand(posargs...; eltype=eltype, celldims=celldims, celltype=celltype, blocklength=blocklength))
end


##
const FALSES_DOC = """
    @falses(args...)
    @falses(args..., <keyword arguments>)

Call `falses(args...)`, where the function `falses` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref).

# Keyword arguments
- `celldims::Integer|NTuple{N,Integer}=1`: the dimensions of each array cell. Each cell can contain a single value (default) or an N-dimensional array of the specified dimensions.
!!! note "Advanced"
    - `blocklength::Integer`: refers to the amount of values of a same `Cell` field that are stored contigously (`blocklength=1` means array of struct like storage; `blocklength=prod(dims)` means array struct of array like storage; `blocklength=0` is an alias for `blocklength=prod(dims)`, enabling better peformance thanks to more specialized dispatch). By default, `blocklength` is automatically set to `0` if a GPU package was chosen with [`@init_parallel_kernel`](@ref) and to `1` if a CPU package was chosen. Furthermore, the argument `blocklength` is only of effect if either `celldims` or `celltype` is set, else it is ignored.

See also: [`@zeros`](@ref), [`@ones`](@ref), [`@rand`](@ref), [`@trues`](@ref), [`@fill`](@ref), [`@CellType`](@ref)
"""
@doc FALSES_DOC
macro falses(args...)
    check_initialized()
    posargs, kwargs_expr = split_args(args)
    celldims, blocklength = extract_kwargvalues(kwargs_expr, (:celldims, :blocklength), "@falses")
    esc(_falses(posargs...; celldims=celldims, blocklength=blocklength))
end


##
const TRUES_DOC = """
    @trues(args...)
    @trues(args..., <keyword arguments>)

Call `trues(args...)`, where the function `trues` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref).

# Keyword arguments
- `celldims::Integer|NTuple{N,Integer}=1`: the dimensions of each array cell. Each cell can contain a single value (default) or an N-dimensional array of the specified dimensions.
!!! note "Advanced"
    - `blocklength::Integer`: refers to the amount of values of a same `Cell` field that are stored contigously (`blocklength=1` means array of struct like storage; `blocklength=prod(dims)` means array struct of array like storage; `blocklength=0` is an alias for `blocklength=prod(dims)`, enabling better peformance thanks to more specialized dispatch). By default, `blocklength` is automatically set to `0` if a GPU package was chosen with [`@init_parallel_kernel`](@ref) and to `1` if a CPU package was chosen. Furthermore, the argument `blocklength` is only of effect if either `celldims` or `celltype` is set, else it is ignored.

See also: [`@zeros`](@ref), [`@ones`](@ref), [`@rand`](@ref), [`@falses`](@ref), [`@fill`](@ref), [`@CellType`](@ref)
"""
@doc TRUES_DOC
macro trues(args...)
    check_initialized()
    posargs, kwargs_expr = split_args(args)
    celldims, blocklength = extract_kwargvalues(kwargs_expr, (:celldims, :blocklength), "@trues")
    esc(_trues(posargs...; celldims=celldims, blocklength=blocklength))
end


##
const FILL_DOC = """
    @fill(x, args...)
    @fill(x, args..., <keyword arguments>)

Call `fill(convert(eltype, x), args...)`, where `eltype` is by default the `numbertype` selected with [`@init_parallel_kernel`](@ref) and the function `fill` is chosen/implemented to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref).

!!! note "Advanced"
    The element type `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Keyword arguments
- `eltype::DataType`: the type of the elements, which can be numbers, booleans or enums.
- `celldims::Integer|NTuple{N,Integer}=1`: the dimensions of each array cell. Each cell can contain a single value (default) or an N-dimensional array of the specified dimensions.
!!! note "Advanced"
    - `celltype::DataType`: the type of each array cell; it must be generated with the macro `@CellType`. The keyword argument `celltype` is incompatible with the other keyword arguments: if any of them is set, then the `celltype` is automatically defined. The `celltype` needs only to be specified to use named cell fields. Note that values can always be addressed with array indices, even when cell field names are defined.
    - `blocklength::Integer`: refers to the amount of values of a same `Cell` field that are stored contigously (`blocklength=1` means array of struct like storage; `blocklength=prod(dims)` means array struct of array like storage; `blocklength=0` is an alias for `blocklength=prod(dims)`, enabling better peformance thanks to more specialized dispatch). By default, `blocklength` is automatically set to `0` if a GPU package was chosen with [`@init_parallel_kernel`](@ref) and to `1` if a CPU package was chosen. Furthermore, the argument `blocklength` is only of effect if either `celldims` or `celltype` is set, else it is ignored.

See also: [`@fill!`](@ref), [`@zeros`](@ref), [`@ones`](@ref), [`@rand`](@ref), [`@falses`](@ref), [`@trues`](@ref), [`@CellType`](@ref)
"""
@doc FILL_DOC
macro fill(args...)
    check_initialized()
    posargs, kwargs_expr = split_args(args)
    eltype, celldims, celltype, blocklength = extract_kwargvalues(kwargs_expr, (:eltype, :celldims, :celltype, :blocklength), "@fill")
    esc(_fill(posargs...; eltype=eltype, celldims=celldims, celltype=celltype, blocklength=blocklength))
end


##
const FILL!_DOC = """
    @fill!(A, x)
    @fill!(A, x)

Call `fill!(A, x)`, where the function `fill` is chosen/implemented to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref).

# Arguments
- `A::Array|CellArray|TArray|TCellArray`: the array to be filled with `x`.
- `x::Number|Enum|Collection{Number|Enum, celldims}`: the content to fill `A` with. If `A` is an CellArray, then `x` can be either a single value or a collection of values (e.g. an array, tuple,...) of the size `celldims` of `A`.

See also: [`@fill`](@ref)
"""
@doc FILL!_DOC
macro fill!(args...) check_initialized(); esc(_fill!(args...)); end


##
const CELLTYPE_DOC = """
    @CellType(name, <keyword arguments>)

Create a cell type, which can then be passed to `@zeros`, `@ones`, `@rand`, `@falses`, `@trues` or `@fill` using the keyword argument `celltype`.

!!! note "Advanced"
    The element type `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory, except if `parametric=true` is set. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Arguments
- `name`: the name of the cell type.

# Keyword arguments
- `eltype::DataType`: the type of the elements, which can be numbers, booleans or enums.
- `fieldnames::String|NTuple{N,String}`: the names of the fields of the cell. Note that cell values can always be addressed with array indices, even when field names are defined.
- `dims::Integer|NTuple{N,Integer}=length(fieldnames)`: the dimensions of the cell. A cell can contain a single value or an N-dimensional array of the specified dimensions. A valid dims argument must fullfill `prod(dims)==length(fieldnames)`. If `dims` is omitted, then it will be automatically set as `dims=length(fieldnames)`, defining the cell to contain a 1-D array of appropriate length.
!!! note "Advanced"
    - `parametric::Bool=false`: whether the cell type has a fixed or parametrisable element type. If `parametric=true` is set, then the keyword argument `eltype` is invalid.

# Examples
    @CellType SymmetricTensor2D fieldnames=(xx, zz, xz)
    #(...)
    A = @zeros(nx, ny, celltype=SymmetricTensor2D)

    @CellType SymmetricTensor3D fieldnames=(xx, yy, zz, yz, xz, xy)
    #(...)
    A = @zeros(nx, ny, nz, celltype=SymmetricTensor3D)

    @CellType Tensor2D fieldnames=(xxxx, yxxx, xyxx, yyxx, xxyx, yxyx, xyyx, yyyx, xxxy, yxxy, xyxy, yyxy, xxyy, yxyy, xyyy, yyyy) dims=(2,2,2,2)
    #(...)
    A = @zeros(nx, ny, celltype=Tensor2D)

    @CellType SymmetricTensor2D fieldnames=(xx, zz, xz) parametric=true
    #(...)
    A = @zeros(nx, ny, celltype=SymmetricTensor2D{Float32})

    @CellType SymmetricTensor2D fieldnames=(xx, zz, xz) eltype=Float32
    #(...)
    A = @zeros(nx, ny, celltype=SymmetricTensor2D)

See also: [`@zeros`](@ref), [`@ones`](@ref), [`@rand`](@ref), [`@falses`](@ref), [`@trues`](@ref), [`@fill`](@ref)
"""
@doc CELLTYPE_DOC
macro CellType(args...)
    check_initialized()
    checkargs_CellType(args...)
    posargs, kwargs_expr = split_args(args)
    eltype, fieldnames, dims, parametric = extract_kwargvalues(kwargs_expr, (:eltype, :fieldnames, :dims, :parametric), "@CellType")
    esc(_CellType(posargs...; eltype=eltype, fieldnames=fieldnames, dims=dims, parametric=parametric))
end


## MACROS FORCING PACKAGE, IGNORING INITIALIZATION

macro zeros_cuda(args...)     check_initialized(); esc(_zeros(args...; package=PKG_CUDA)); end
macro ones_cuda(args...)      check_initialized(); esc(_ones(args...; package=PKG_CUDA)); end
macro rand_cuda(args...)      check_initialized(); esc(_rand(args...; package=PKG_CUDA)); end
macro falses_cuda(args...)    check_initialized(); esc(_falses(args...; package=PKG_CUDA)); end
macro trues_cuda(args...)     check_initialized(); esc(_trues(args...; package=PKG_CUDA)); end
macro fill_cuda(args...)      check_initialized(); esc(_fill(args...; package=PKG_CUDA)); end
macro fill!_cuda(args...)     check_initialized(); esc(_fill!(args...; package=PKG_CUDA)); end
macro zeros_amdgpu(args...)     check_initialized(); esc(_zeros(args...; package=PKG_AMDGPU)); end
macro ones_amdgpu(args...)      check_initialized(); esc(_ones(args...; package=PKG_AMDGPU)); end
macro rand_amdgpu(args...)      check_initialized(); esc(_rand(args...; package=PKG_AMDGPU)); end
macro falses_amdgpu(args...)    check_initialized(); esc(_falses(args...; package=PKG_AMDGPU)); end
macro trues_amdgpu(args...)     check_initialized(); esc(_trues(args...; package=PKG_AMDGPU)); end
macro fill_amdgpu(args...)      check_initialized(); esc(_fill(args...; package=PKG_AMDGPU)); end
macro fill!_amdgpu(args...)     check_initialized(); esc(_fill!(args...; package=PKG_AMDGPU)); end
macro zeros_threads(args...)  check_initialized(); esc(_zeros(args...; package=PKG_THREADS)); end
macro ones_threads(args...)   check_initialized(); esc(_ones(args...; package=PKG_THREADS)); end
macro rand_threads(args...)   check_initialized(); esc(_rand(args...; package=PKG_THREADS)); end
macro falses_threads(args...) check_initialized(); esc(_falses(args...; package=PKG_THREADS)); end
macro trues_threads(args...)  check_initialized(); esc(_trues(args...; package=PKG_THREADS)); end
macro fill_threads(args...)   check_initialized(); esc(_fill(args...; package=PKG_THREADS)); end
macro fill!_threads(args...)  check_initialized(); esc(_fill!(args...; package=PKG_THREADS)); end


## ARGUMENT CHECKS

function checkargs_CellType(args...)
    if isempty(args) @ArgumentError("arguments missing.") end
    posargs, kwargs_expr = split_args(args)
    if length(posargs) != 1 @ArgumentError("exactly one positional argument is required.") end
    if length(kwargs_expr) < 1 @ArgumentError("the fieldnames keyword argument is mandatory.") end
    if length(kwargs_expr) > 4 @ArgumentError("too many keyword arguments.") end
end


## ALLOCATOR FUNCTIONS

function _zeros(args...; eltype=nothing, celldims=nothing, celltype=nothing, blocklength=nothing, package::Symbol=get_package())
    celltype    = determine_celltype(eltype, celldims, celltype)
    blocklength = determine_blocklength(blocklength, package)
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.zeros_cuda($celltype, $blocklength, $(args...)))
    elseif (package == PKG_AMDGPU)  return :(ParallelStencil.ParallelKernel.zeros_amdgpu($celltype, $blocklength, $(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.zeros_cpu($celltype, $blocklength, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _ones(args...; eltype=nothing, celldims=nothing, celltype=nothing, blocklength=nothing, package::Symbol=get_package())
    celltype    = determine_celltype(eltype, celldims, celltype)
    blocklength = determine_blocklength(blocklength, package)
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.ones_cuda($celltype, $blocklength, $(args...)))
    elseif (package == PKG_AMDGPU)  return :(ParallelStencil.ParallelKernel.ones_amdgpu($celltype, $blocklength, $(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.ones_cpu($celltype, $blocklength, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _rand(args...; eltype=nothing, celldims=nothing, celltype=nothing, blocklength=nothing, package::Symbol=get_package())
    celltype    = determine_celltype(eltype, celldims, celltype)
    blocklength = determine_blocklength(blocklength, package)
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.rand_cuda($celltype, $blocklength, $(args...)))
    elseif (package == PKG_AMDGPU)  return :(ParallelStencil.ParallelKernel.rand_amdgpu($celltype, $blocklength, $(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.rand_cpu($celltype, $blocklength, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _falses(args...; celldims=nothing, blocklength=nothing, package::Symbol=get_package())
    celltype    = determine_celltype(Bool, celldims, nothing)
    blocklength = determine_blocklength(blocklength, package)
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.falses_cuda($celltype, $blocklength, $(args...)))
    elseif (package == PKG_AMDGPU)  return :(ParallelStencil.ParallelKernel.falses_amdgpu($celltype, $blocklength, $(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.falses_cpu($celltype, $blocklength, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _trues(args...; celldims=nothing, blocklength=nothing, package::Symbol=get_package())
    celltype    = determine_celltype(Bool, celldims, nothing)
    blocklength = determine_blocklength(blocklength, package)
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.trues_cuda($celltype, $blocklength, $(args...)))
    elseif (package == PKG_AMDGPU)  return :(ParallelStencil.ParallelKernel.trues_amdgpu($celltype, $blocklength, $(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.trues_cpu($celltype, $blocklength, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _fill(args...; eltype=nothing, celldims=nothing, celltype=nothing, blocklength=nothing, package::Symbol=get_package())
    celltype    = determine_celltype(eltype, celldims, celltype)
    blocklength = determine_blocklength(blocklength, package)
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.fill_cuda($celltype, $blocklength, $(args...)))
    elseif (package == PKG_AMDGPU)  return :(ParallelStencil.ParallelKernel.fill_amdgpu($celltype, $blocklength, $(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.fill_cpu($celltype, $blocklength, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _fill!(args...; package::Symbol=get_package())
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.fill_cuda!($(args...)))
    elseif (package == PKG_AMDGPU)  return :(ParallelStencil.ParallelKernel.fill_amdgpu!($(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.fill_cpu!($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _CellType(name; eltype=nothing, fieldnames=nothing, dims=nothing, parametric=nothing)
    if isnothing(fieldnames) @ArgumentError("@CellType: the keyword argument 'fieldnames' is mandatory.") end
    fieldnames = parse_kwargvalues(fieldnames)
    if isnothing(dims)
        dims = [length(fieldnames)]
    else
        dims = parse_kwargvalues(dims)
    end
    if isnothing(parametric) parametric=false end
    if isnothing(eltype)
        eltype = get_numbertype()
        if (!parametric && eltype == NUMBERTYPE_NONE) @ArgumentError("@CellType: the keyword argument 'eltype' is mandatory when no default is set (and the keyword argument `parametric=true` not set).") end
    else
        if (parametric) @ArgumentError("@CellType: the keyword argument 'eltype' is invalid when `parametric=true` is set.") end
    end
    fields = Expr[]
    if parametric
        for fieldname in fieldnames
            push!(fields, quote $fieldname::T end)
        end
        quote
            struct $name{T} <: ParallelStencil.ParallelKernel.FieldArray{Tuple{$(dims...)}, T, length($dims)}
                $(fields...)
            end
        end
    else
        for fieldname in fieldnames
            push!(fields, quote $fieldname::$eltype end)
        end
        quote
            struct $name <: ParallelStencil.ParallelKernel.FieldArray{Tuple{$(dims...)}, $eltype, length($dims)}
                $(fields...)
            end
        end
    end
end

is_symbol(arg) = isa(arg, Symbol)
is_tuple(arg)  = isa(arg, Expr) && (arg.head==:tuple)

function parse_kwargvalues(arg)
    if    is_symbol(arg) return [arg]
    elseif is_tuple(arg) return arg.args
    else                 @ArgumentError("@CellType: the keyword argument value $arg is not valid. ")
    end
end

function determine_celltype(eltype, celldims, celltype)
    if !isnothing(celltype)
        if (!isnothing(celldims) || !isnothing(eltype)) @ArgumentError("the keyword argument 'celltype' is incompatible with the other keyword arguments.") end
    else
        if isnothing(eltype)
            eltype = get_numbertype()
            if (eltype == NUMBERTYPE_NONE) @ArgumentError("the keyword argument 'eltype' is mandatory in @zeros, @ones, @rand and @fill when no default is set.") end
        end
        if !isnothing(celldims) celltype = :(ParallelStencil.ParallelKernel.SArray{Tuple{$celldims...}, $eltype, length($celldims), prod($celldims)})
        else                    celltype = eltype
        end
    end
    return celltype
end

function determine_blocklength(blocklength, package)
    if isnothing(blocklength) blocklength = CELLARRAY_BLOCKLENGTH[package] end
    return Val(blocklength)
end


## RUNTIME ALLOCATOR FUNCTIONS 

 zeros_cpu(::Type{T}, blocklength, args...) where {T<:Number}                      = (check_datatype(T); Base.zeros(T, args...))                # (blocklength is ignored if neither celldims nor celltype is set)
  ones_cpu(::Type{T}, blocklength, args...) where {T<:Number}                      = (check_datatype(T); fill_cpu(T, blocklength, 1, args...))  # ...
  rand_cpu(::Type{T}, blocklength, args...) where {T<:Union{Number,Enum}}          = (check_datatype(T, Bool, Enum); Base.rand(T, args...))     # ...
falses_cpu(::Type{T}, blocklength, args...) where {T<:Bool}                        = Base.falses(args...)                                       # ...
 trues_cpu(::Type{T}, blocklength, args...) where {T<:Bool}                        = Base.trues(args...)                                        # ...  #Note: an alternative would be: fill_cpu(T, true, args...)

 zeros_cpu(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = (check_datatype(T); fill_cpu(T, blocklength, 0, args...))
  ones_cpu(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = (check_datatype(T); fill_cpu(T, blocklength, 1, args...))
  rand_cpu(::Type{T}, ::Val{B},    dims)    where {T<:Union{SArray,FieldArray}, B} = (check_datatype(T, Bool, Enum); blocklen = (B == 0) ? prod(dims) : B; CellArray{T,length(dims),B}(Base.rand(eltype(T), blocklen, prod(size(T)), ceil(Int,prod(dims)/blocklen)), dims))
  rand_cpu(::Type{T}, blocklength, dims...) where {T<:Union{SArray,FieldArray}}    = rand_cpu(T, blocklength, dims)
falses_cpu(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = fill_cpu(T, blocklength, false, args...)
 trues_cpu(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = fill_cpu(T, blocklength, true, args...)

 zeros_cuda(::Type{T}, blocklength, args...) where {T<:Number}                      = (check_datatype(T); CUDA.zeros(T, args...))  # (blocklength is ignored if neither celldims nor celltype is set)
  ones_cuda(::Type{T}, blocklength, args...) where {T<:Number}                      = (check_datatype(T); CUDA.ones(T, args...))   # ...
  rand_cuda(::Type{T}, blocklength, args...) where {T<:Union{Number,Enum}}          = CuArray(rand_cpu(T, blocklength, args...))   # ...
falses_cuda(::Type{T}, blocklength, args...) where {T<:Bool}                        = CUDA.zeros(Bool, args...)                         # ...
 trues_cuda(::Type{T}, blocklength, args...) where {T<:Bool}                        = CUDA.ones(Bool, args...)                          # ...
  fill_cuda(::Type{T}, blocklength, args...) where {T<:Union{Number,Enum}}          = CuArray(fill_cpu(T, blocklength, args...))   # ...

 zeros_cuda(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = (check_datatype(T); fill_cuda(T, blocklength, 0, args...))
  ones_cuda(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = (check_datatype(T); fill_cuda(T, blocklength, 1, args...))
  rand_cuda(::Type{T}, ::Val{B},    dims)    where {T<:Union{SArray,FieldArray}, B} = (check_datatype(T, Bool, Enum); blocklen = (B == 0) ? prod(dims) : B; CellArray{T,length(dims),B, CUDA.CuArray{eltype(T),3}}(CUDA.rand(eltype(T), blocklen, prod(size(T)), ceil(Int,prod(dims)/(blocklen))), dims))
  rand_cuda(::Type{T}, blocklength, dims...) where {T<:Union{SArray,FieldArray}}    = rand_cuda(T, blocklength, dims)
falses_cuda(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = fill_cuda(T, blocklength, false, args...)
 trues_cuda(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = fill_cuda(T, blocklength, true, args...)

 zeros_amdgpu(::Type{T}, blocklength, args...) where {T<:Number}                      = (check_datatype(T); AMDGPU.zeros(T, args...))  # (blocklength is ignored if neither celldims nor celltype is set)
  ones_amdgpu(::Type{T}, blocklength, args...) where {T<:Number}                      = (check_datatype(T); AMDGPU.ones(T, args...))   # ...
  rand_amdgpu(::Type{T}, blocklength, args...) where {T<:Union{Number,Enum}}          = ROCArray(rand_cpu(T, blocklength, args...))   # ...
falses_amdgpu(::Type{T}, blocklength, args...) where {T<:Bool}                        = AMDGPU.falses(args...)                         # ...
 trues_amdgpu(::Type{T}, blocklength, args...) where {T<:Bool}                        = AMDGPU.trues(args...)                          # ...
  fill_amdgpu(::Type{T}, blocklength, args...) where {T<:Union{Number,Enum}}          = ROCArray(fill_cpu(T, blocklength, args...))   # ...

 zeros_amdgpu(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = (check_datatype(T); fill_amdgpu(T, blocklength, 0, args...))
  ones_amdgpu(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = (check_datatype(T); fill_amdgpu(T, blocklength, 1, args...))
  rand_amdgpu(::Type{T}, ::Val{B},    dims)    where {T<:Union{SArray,FieldArray}, B} = (check_datatype(T, Bool, Enum); blocklen = (B == 0) ? prod(dims) : B; CellArray{T,length(dims),B, AMDGPU.ROCArray{eltype(T),3}}(AMDGPU.rand(eltype(T), blocklen, prod(size(T)), ceil(Int,prod(dims)/(blocklen))), dims))
  rand_amdgpu(::Type{T}, blocklength, dims...) where {T<:Union{SArray,FieldArray}}    = rand_amdgpu(T, blocklength, dims)
falses_amdgpu(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = fill_amdgpu(T, blocklength, false, args...)
 trues_amdgpu(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = fill_amdgpu(T, blocklength, true, args...)

function fill_cpu(::Type{T}, blocklength, x, args...) where {T <: Union{Number, Enum}} # (blocklength is ignored if neither celldims nor celltype is set)
    if (!(eltype(x) <: Number) || (eltype(x) == Bool)) && (eltype(x) != eltype(T)) @ArgumentError("fill: the (element) type of argument 'x' is not a normal number type ($(eltype(x))), but does not match the obtained (default) 'eltype' ($(eltype(T))); automatic conversion to $(eltype(T)) is therefore not attempted. Set the keyword argument 'eltype' accordingly to the element type of 'x' or pass an 'x' of a different (element) type.") end
    check_datatype(T, Bool, Enum)
    cell = convert(T, x)
    return Base.fill(cell, args...)
end

function fill_cpu(::Type{T}, ::Val{B}, x, args...) where {T <: Union{SArray,FieldArray}, B}
    if (!(eltype(x) <: Number) || (eltype(x) == Bool)) && (eltype(x) != eltype(T)) @ArgumentError("fill: the (element) type of argument 'x' is not a normal number type ($(eltype(x))), but does not match the obtained (default) 'eltype' ($(eltype(T))); automatic conversion to $(eltype(T)) is therefore not attempted. Set the keyword argument 'eltype' accordingly to the element type of 'x' or pass an 'x' of a different (element) type.") end
    check_datatype(T, Bool, Enum)
    if     (length(x) == 1)         cell = convert(T, fill(convert(eltype(T), x), size(T)))
    elseif (length(x) == length(T)) cell = convert(T, x)
    else                            @ArgumentError("fill: argument 'x' contains the wrong number of elements ($(length(x))). It must be a scalar or contain the number of elements defined by 'celldims'.")
    end
    return CellArrays.fill!(CPUCellArray{T,B}(undef, args...), cell)
end

function fill_cuda(::Type{T}, ::Val{B}, x, args...) where {T <: Union{SArray,FieldArray}, B}
    if (!(eltype(x) <: Number) || (eltype(x) == Bool)) && (eltype(x) != eltype(T)) @ArgumentError("fill: the (element) type of argument 'x' is not a normal number type ($(eltype(x))), but does not match the obtained (default) 'eltype' ($(eltype(T))); automatic conversion to $(eltype(T)) is therefore not attempted. Set the keyword argument 'eltype' accordingly to the element type of 'x' or pass an 'x' of a different (element) type.") end
    check_datatype(T, Bool, Enum)
    if     (length(x) == 1)         cell = convert(T, fill(convert(eltype(T), x), size(T)))
    elseif (length(x) == length(T)) cell = convert(T, x)
    else                            @ArgumentError("fill: argument 'x' contains the wrong number of elements ($(length(x))). It must be a scalar or contain the number of elements defined by 'celldims'.")
    end
    return CellArrays.fill!(CuCellArray{T,B}(undef, args...), cell)
end

function fill_amdgpu(::Type{T}, ::Val{B}, x, args...) where {T <: Union{SArray,FieldArray}, B}
    if (!(eltype(x) <: Number) || (eltype(x) == Bool)) && (eltype(x) != eltype(T)) @ArgumentError("fill: the (element) type of argument 'x' is not a normal number type ($(eltype(x))), but does not match the obtained (default) 'eltype' ($(eltype(T))); automatic conversion to $(eltype(T)) is therefore not attempted. Set the keyword argument 'eltype' accordingly to the element type of 'x' or pass an 'x' of a different (element) type.") end
    check_datatype(T, Bool, Enum)
    if     (length(x) == 1)         cell = convert(T, fill(convert(eltype(T), x), size(T)))
    elseif (length(x) == length(T)) cell = convert(T, x)
    else                            @ArgumentError("fill: argument 'x' contains the wrong number of elements ($(length(x))). It must be a scalar or contain the number of elements defined by 'celldims'.")
    end
    return CellArrays.fill!(ROCCellArray{T,B}(undef, args...), cell)
end

fill_cpu!(A, x) = Base.fill!(A, construct_cell(A, x))
fill_cuda!(A, x) = CUDA.fill!(A, construct_cell(A, x))
fill_amdgpu!(A, x) = AMDGPU.fill!(A, construct_cell(A, x))

#TODO: eliminate nearly duplicate code starting from if T_cell I think...
function construct_cell(A, x)
    T_cell = eltype(A)
    if (!(eltype(x) <: Number) || (eltype(x) == Bool)) && (eltype(x) != eltype(T_cell)) @ArgumentError("@fill!: the (element) type of argument 'x' is not a normal number type ($(eltype(x))), but does not match the type of the elements of 'A' ($(eltype(T_cell))); automatic conversion to $(eltype(T_cell)) is therefore not attempted. Pass an 'x' of a different (element) type.") end
    if T_cell <: Union{SArray,FieldArray}
        if     (length(x) == 1)              cell = convert(T_cell, fill(convert(eltype(T_cell), x), size(T_cell)))
        elseif (length(x) == length(T_cell)) cell = convert(T_cell, x)
        else                                 @ArgumentError("@fill!: argument 'x' contains the wrong number of elements ($(length(x))). It must be a scalar or contain the number of elements defined by 'celldims' or 'celltype'.")
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
eltype(::Type{T}) where {T <:Enum} = T

import Random
Random.SamplerType{T}() where {T<:Enum} = Random.Sampler(Random.GLOBAL_RNG, instances(T))
