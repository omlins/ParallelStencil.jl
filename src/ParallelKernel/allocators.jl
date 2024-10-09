##
const ZEROS_DOC = """
    @zeros(args...)
    @zeros(args..., <keyword arguments>)

Call `zeros(eltype, args...)`, where `eltype` is by default the `numbertype` selected with [`@init_parallel_kernel`](@ref) and the function `zeros` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (zeros for Threads or Polyester, CUDA.zeros for CUDA, AMDGPU.zeros for AMDGPU and Metal.zeros for Metal).

!!! note "Advanced"
    The `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Keyword arguments
- `eltype::DataType`: the type of the elements (numbers or indices).
- `celldims::Integer|NTuple{N,Integer}=1`: the dimensions of each array cell. Each cell can contain a single value (default) or an N-dimensional array of the specified dimensions.
!!! note "Advanced"
    - `celltype::DataType`: the type of each array cell; it must be generated with the macro `@CellType`. The keyword argument `celltype` is incompatible with the other keyword arguments: if any of them is set, then the `celltype` is automatically defined. The `celltype` needs only to be specified to use named cell fields. Note that values can always be addressed with array indices, even when cell field names are defined.
    - `blocklength::Integer`: refers to the amount of values of a same `Cell` field that are stored contigously (`blocklength=1` means array of struct like storage; `blocklength=prod(dims)` means array struct of array like storage; `blocklength=0` is an alias for `blocklength=prod(dims)`, enabling better peformance thanks to more specialized dispatch). By default, `blocklength` is automatically set to `0` if a GPU package was chosen with [`@init_parallel_kernel`](@ref) and to `1` if a CPU package was chosen. Furthermore, the argument `blocklength` is only of effect if either `celldims` or `celltype` is set, else it is ignored.

See also: [`@ones`](@ref), [`@rand`](@ref), [`@falses`](@ref), [`@trues`](@ref), [`@fill`](@ref), [`@CellType`](@ref)
"""
@doc ZEROS_DOC
macro zeros(args...)
    check_initialized(__module__)
    posargs, kwargs_expr = split_args(args)
    eltype, celldims, celltype, blocklength = extract_kwargvalues(kwargs_expr, (:eltype, :celldims, :celltype, :blocklength), "@zeros")
    esc(_zeros(__module__, posargs...; eltype=eltype, celldims=celldims, celltype=celltype, blocklength=blocklength))
end


##
const ONES_DOC = """
    @ones(args...)
    @ones(args..., <keyword arguments>)

Call `ones(eltype, args...)`, where `eltype` is by default the `numbertype` selected with [`@init_parallel_kernel`](@ref) and the function `ones` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (ones for Threads or Polyester, CUDA.ones for CUDA, AMDGPU.ones for AMDGPU and Metal.ones for Metal).

!!! note "Advanced"
    The `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Keyword arguments
- `eltype::DataType`: the type of the elements (numbers or indices).
- `celldims::Integer|NTuple{N,Integer}=1`: the dimensions of each array cell. Each cell can contain a single value (default) or an N-dimensional array of the specified dimensions.
!!! note "Advanced"
    - `celltype::DataType`: the type of each array cell; it must be generated with the macro `@CellType`. The keyword argument `celltype` is incompatible with the other keyword arguments: if any of them is set, then the `celltype` is automatically defined. The `celltype` needs only to be specified to use named cell fields. Note that values can always be addressed with array indices, even when cell field names are defined.
    - `blocklength::Integer`: refers to the amount of values of a same `Cell` field that are stored contigously (`blocklength=1` means array of struct like storage; `blocklength=prod(dims)` means array struct of array like storage; `blocklength=0` is an alias for `blocklength=prod(dims)`, enabling better peformance thanks to more specialized dispatch). By default, `blocklength` is automatically set to `0` if a GPU package was chosen with [`@init_parallel_kernel`](@ref) and to `1` if a CPU package was chosen. Furthermore, the argument `blocklength` is only of effect if either `celldims` or `celltype` is set, else it is ignored.

See also: [`@zeros`](@ref), [`@rand`](@ref), [`@falses`](@ref), [`@trues`](@ref), [`@fill`](@ref), [`@CellType`](@ref)
"""
@doc ONES_DOC
macro ones(args...)
    check_initialized(__module__)
    posargs, kwargs_expr = split_args(args)
    eltype, celldims, celltype, blocklength = extract_kwargvalues(kwargs_expr, (:eltype, :celldims, :celltype, :blocklength), "@ones")
    esc(_ones(__module__, posargs...; eltype=eltype, celldims=celldims, celltype=celltype, blocklength=blocklength))
end

##
const RAND_DOC = """
    @rand(args...)
    @rand(args..., <keyword arguments>)

Call `rand(eltype, args...)`, where `eltype` is by default the `numbertype` selected with [`@init_parallel_kernel`](@ref) and the function `rand` is chosen/implemented to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref).

!!! note "Advanced"
    The `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Keyword arguments
- `eltype::DataType`: the type of the elements, which can be numbers, indices, booleans or enums.
- `celldims::Integer|NTuple{N,Integer}=1`: the dimensions of each array cell. Each cell can contain a single value (default) or an N-dimensional array of the specified dimensions.
!!! note "Advanced"
    - `celltype::DataType`: the type of each array cell; it must be generated with the macro `@CellType`. The keyword argument `celltype` is incompatible with the other keyword arguments: if any of them is set, then the `celltype` is automatically defined. The `celltype` needs only to be specified to use named cell fields. Note that values can always be addressed with array indices, even when cell field names are defined.
    - `blocklength::Integer`: refers to the amount of values of a same `Cell` field that are stored contigously (`blocklength=1` means array of struct like storage; `blocklength=prod(dims)` means array struct of array like storage; `blocklength=0` is an alias for `blocklength=prod(dims)`, enabling better peformance thanks to more specialized dispatch). By default, `blocklength` is automatically set to `0` if a GPU package was chosen with [`@init_parallel_kernel`](@ref) and to `1` if a CPU package was chosen. Furthermore, the argument `blocklength` is only of effect if either `celldims` or `celltype` is set, else it is ignored.

See also: [`@zeros`](@ref), [`@ones`](@ref), [`@falses`](@ref), [`@trues`](@ref), [`@fill`](@ref), [`@CellType`](@ref)
"""
@doc RAND_DOC
macro rand(args...)
    check_initialized(__module__)
    posargs, kwargs_expr = split_args(args)
    eltype, celldims, celltype, blocklength = extract_kwargvalues(kwargs_expr, (:eltype, :celldims, :celltype, :blocklength), "@rand")
    esc(_rand(__module__, posargs...; eltype=eltype, celldims=celldims, celltype=celltype, blocklength=blocklength))
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
    check_initialized(__module__)
    posargs, kwargs_expr = split_args(args)
    celldims, blocklength = extract_kwargvalues(kwargs_expr, (:celldims, :blocklength), "@falses")
    esc(_falses(__module__, posargs...; celldims=celldims, blocklength=blocklength))
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
    check_initialized(__module__)
    posargs, kwargs_expr = split_args(args)
    celldims, blocklength = extract_kwargvalues(kwargs_expr, (:celldims, :blocklength), "@trues")
    esc(_trues(__module__, posargs...; celldims=celldims, blocklength=blocklength))
end


##
const FILL_DOC = """
    @fill(x, args...)
    @fill(x, args..., <keyword arguments>)

Call `fill(convert(eltype, x), args...)`, where `eltype` is by default the `numbertype` selected with [`@init_parallel_kernel`](@ref) and the function `fill` is chosen/implemented to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref).

!!! note "Advanced"
    The element type `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Keyword arguments
- `eltype::DataType`: the type of the elements, which can be numbers, indices, booleans or enums.
- `celldims::Integer|NTuple{N,Integer}=1`: the dimensions of each array cell. Each cell can contain a single value (default) or an N-dimensional array of the specified dimensions.
!!! note "Advanced"
    - `celltype::DataType`: the type of each array cell; it must be generated with the macro `@CellType`. The keyword argument `celltype` is incompatible with the other keyword arguments: if any of them is set, then the `celltype` is automatically defined. The `celltype` needs only to be specified to use named cell fields. Note that values can always be addressed with array indices, even when cell field names are defined.
    - `blocklength::Integer`: refers to the amount of values of a same `Cell` field that are stored contigously (`blocklength=1` means array of struct like storage; `blocklength=prod(dims)` means array struct of array like storage; `blocklength=0` is an alias for `blocklength=prod(dims)`, enabling better peformance thanks to more specialized dispatch). By default, `blocklength` is automatically set to `0` if a GPU package was chosen with [`@init_parallel_kernel`](@ref) and to `1` if a CPU package was chosen. Furthermore, the argument `blocklength` is only of effect if either `celldims` or `celltype` is set, else it is ignored.

See also: [`@fill!`](@ref), [`@zeros`](@ref), [`@ones`](@ref), [`@rand`](@ref), [`@falses`](@ref), [`@trues`](@ref), [`@CellType`](@ref)
"""
@doc FILL_DOC
macro fill(args...)
    check_initialized(__module__)
    posargs, kwargs_expr = split_args(args)
    eltype, celldims, celltype, blocklength = extract_kwargvalues(kwargs_expr, (:eltype, :celldims, :celltype, :blocklength), "@fill")
    esc(_fill(__module__, posargs...; eltype=eltype, celldims=celldims, celltype=celltype, blocklength=blocklength))
end


##
const FILL!_DOC = """
    @fill!(A, x)
    @fill!(A, x)

Call `fill!(A, x)`, where the function `fill` is chosen/implemented to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref).

# Arguments
- `A::Array|CellArray|TArray|TCellArray`: the array to be filled with `x`.
- `x::Number|Data.Index|Enum|Collection{Number|Data.Index|Enum, celldims}`: the content to fill `A` with. If `A` is an CellArray, then `x` can be either a single value or a collection of values (e.g. an array, tuple,...) of the size `celldims` of `A`.

See also: [`@fill`](@ref)
"""
@doc FILL!_DOC
macro fill!(args...) check_initialized(__module__); esc(_fill!(__module__, args...)); end


##
const CELLTYPE_DOC = """
    @CellType(name, <keyword arguments>)

Create a cell type, which can then be passed to `@zeros`, `@ones`, `@rand`, `@falses`, `@trues` or `@fill` using the keyword argument `celltype`.

!!! note "Advanced"
    The element type `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory, except if `parametric=true` is set. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Arguments
- `name`: the name of the cell type.

# Keyword arguments
- `eltype::DataType`: the type of the elements, which can be numbers, indices, booleans or enums.
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
    check_initialized(__module__)
    checkargs_CellType(args...)
    posargs, kwargs_expr = split_args(args)
    eltype, fieldnames, dims, parametric = extract_kwargvalues(kwargs_expr, (:eltype, :fieldnames, :dims, :parametric), "@CellType")
    esc(_CellType(__module__, posargs...; eltype=eltype, fieldnames=fieldnames, dims=dims, parametric=parametric))
end


## MACROS FORCING PACKAGE, IGNORING INITIALIZATION

macro zeros_cuda(args...)       check_initialized(__module__); esc(_zeros(__module__, args...; package=PKG_CUDA)); end
macro ones_cuda(args...)        check_initialized(__module__); esc(_ones(__module__, args...; package=PKG_CUDA)); end
macro rand_cuda(args...)        check_initialized(__module__); esc(_rand(__module__, args...; package=PKG_CUDA)); end
macro falses_cuda(args...)      check_initialized(__module__); esc(_falses(__module__, args...; package=PKG_CUDA)); end
macro trues_cuda(args...)       check_initialized(__module__); esc(_trues(__module__, args...; package=PKG_CUDA)); end
macro fill_cuda(args...)        check_initialized(__module__); esc(_fill(__module__, args...; package=PKG_CUDA)); end
macro fill!_cuda(args...)       check_initialized(__module__); esc(_fill!(__module__, args...; package=PKG_CUDA)); end
macro zeros_amdgpu(args...)     check_initialized(__module__); esc(_zeros(__module__, args...; package=PKG_AMDGPU)); end
macro ones_amdgpu(args...)      check_initialized(__module__); esc(_ones(__module__, args...; package=PKG_AMDGPU)); end
macro rand_amdgpu(args...)      check_initialized(__module__); esc(_rand(__module__, args...; package=PKG_AMDGPU)); end
macro falses_amdgpu(args...)    check_initialized(__module__); esc(_falses(__module__, args...; package=PKG_AMDGPU)); end
macro trues_amdgpu(args...)     check_initialized(__module__); esc(_trues(__module__, args...; package=PKG_AMDGPU)); end
macro fill_amdgpu(args...)      check_initialized(__module__); esc(_fill(__module__, args...; package=PKG_AMDGPU)); end
macro fill!_amdgpu(args...)     check_initialized(__module__); esc(_fill!(__module__, args...; package=PKG_AMDGPU)); end
macro zeros_metal(args...)      check_initialized(__module__); esc(_zeros(__module__, args...; package=PKG_METAL)); end
macro ones_metal(args...)       check_initialized(__module__); esc(_ones(__module__, args...; package=PKG_METAL)); end
macro rand_metal(args...)       check_initialized(__module__); esc(_rand(__module__, args...; package=PKG_METAL)); end
macro falses_metal(args...)     check_initialized(__module__); esc(_falses(__module__, args...; package=PKG_METAL)); end
macro trues_metal(args...)      check_initialized(__module__); esc(_trues(__module__, args...; package=PKG_METAL)); end
macro fill_metal(args...)       check_initialized(__module__); esc(_fill(__module__, args...; package=PKG_METAL)); end
macro fill!_metal(args...)      check_initialized(__module__); esc(_fill!(__module__, args...; package=PKG_METAL)); end
macro zeros_threads(args...)    check_initialized(__module__); esc(_zeros(__module__, args...; package=PKG_THREADS)); end
macro ones_threads(args...)     check_initialized(__module__); esc(_ones(__module__, args...; package=PKG_THREADS)); end
macro rand_threads(args...)     check_initialized(__module__); esc(_rand(__module__, args...; package=PKG_THREADS)); end
macro falses_threads(args...)   check_initialized(__module__); esc(_falses(__module__, args...; package=PKG_THREADS)); end
macro trues_threads(args...)    check_initialized(__module__); esc(_trues(__module__, args...; package=PKG_THREADS)); end
macro fill_threads(args...)     check_initialized(__module__); esc(_fill(__module__, args...; package=PKG_THREADS)); end
macro fill!_threads(args...)    check_initialized(__module__); esc(_fill!(__module__, args...; package=PKG_THREADS)); end
macro zeros_polyester(args...)  check_initialized(__module__); esc(_zeros(__module__, args...; package=PKG_POLYESTER)); end
macro ones_polyester(args...)   check_initialized(__module__); esc(_ones(__module__, args...; package=PKG_POLYESTER)); end
macro rand_polyester(args...)   check_initialized(__module__); esc(_rand(__module__, args...; package=PKG_POLYESTER)); end
macro falses_polyester(args...) check_initialized(__module__); esc(_falses(__module__, args...; package=PKG_POLYESTER)); end
macro trues_polyester(args...)  check_initialized(__module__); esc(_trues(__module__, args...; package=PKG_POLYESTER)); end
macro fill_polyester(args...)   check_initialized(__module__); esc(_fill(__module__, args...; package=PKG_POLYESTER)); end
macro fill!_polyester(args...)  check_initialized(__module__); esc(_fill!(__module__, args...; package=PKG_POLYESTER)); end


## ARGUMENT CHECKS

function checkargs_CellType(args...)
    if isempty(args) @ArgumentError("arguments missing.") end
    posargs, kwargs_expr = split_args(args)
    if length(posargs) != 1 @ArgumentError("exactly one positional argument is required.") end
    if length(kwargs_expr) < 1 @ArgumentError("the fieldnames keyword argument is mandatory.") end
    if length(kwargs_expr) > 4 @ArgumentError("too many keyword arguments.") end
end


## ALLOCATOR FUNCTIONS

function _zeros(caller::Module, args...; eltype=nothing, celldims=nothing, celltype=nothing, blocklength=nothing, package::Symbol=get_package(caller))
    celltype    = determine_celltype(caller, eltype, celldims, celltype)
    blocklength = determine_blocklength(blocklength, package)
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.zeros_cuda($celltype, $blocklength, $(args...)))
    elseif (package == PKG_AMDGPU)  return :(ParallelStencil.ParallelKernel.zeros_amdgpu($celltype, $blocklength, $(args...)))
    elseif (package == PKG_METAL)   return :(ParallelStencil.ParallelKernel.zeros_metal($celltype, $blocklength, $(args...)))
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.zeros_cpu($celltype, $blocklength, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _ones(caller::Module, args...; eltype=nothing, celldims=nothing, celltype=nothing, blocklength=nothing, package::Symbol=get_package(caller))
    celltype    = determine_celltype(caller, eltype, celldims, celltype)
    blocklength = determine_blocklength(blocklength, package)
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.ones_cuda($celltype, $blocklength, $(args...)))
    elseif (package == PKG_AMDGPU)  return :(ParallelStencil.ParallelKernel.ones_amdgpu($celltype, $blocklength, $(args...)))
    elseif (package == PKG_METAL)   return :(ParallelStencil.ParallelKernel.ones_metal($celltype, $blocklength, $(args...)))
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.ones_cpu($celltype, $blocklength, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _rand(caller::Module, args...; eltype=nothing, celldims=nothing, celltype=nothing, blocklength=nothing, package::Symbol=get_package(caller))
    celltype    = determine_celltype(caller, eltype, celldims, celltype)
    blocklength = determine_blocklength(blocklength, package)
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.rand_cuda($celltype, $blocklength, $(args...)))
    elseif (package == PKG_AMDGPU)  return :(ParallelStencil.ParallelKernel.rand_amdgpu($celltype, $blocklength, $(args...)))
    elseif (package == PKG_METAL)   return :(ParallelStencil.ParallelKernel.rand_metal($celltype, $blocklength, $(args...)))
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.rand_cpu($celltype, $blocklength, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _falses(caller::Module, args...; celldims=nothing, blocklength=nothing, package::Symbol=get_package(caller))
    celltype    = determine_celltype(caller, Bool, celldims, nothing)
    blocklength = determine_blocklength(blocklength, package)
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.falses_cuda($celltype, $blocklength, $(args...)))
    elseif (package == PKG_AMDGPU)  return :(ParallelStencil.ParallelKernel.falses_amdgpu($celltype, $blocklength, $(args...)))
    elseif (package == PKG_METAL)   return :(ParallelStencil.ParallelKernel.falses_metal($celltype, $blocklength, $(args...)))
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.falses_cpu($celltype, $blocklength, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _trues(caller::Module, args...; celldims=nothing, blocklength=nothing, package::Symbol=get_package(caller))
    celltype    = determine_celltype(caller, Bool, celldims, nothing)
    blocklength = determine_blocklength(blocklength, package)
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.trues_cuda($celltype, $blocklength, $(args...)))
    elseif (package == PKG_AMDGPU)  return :(ParallelStencil.ParallelKernel.trues_amdgpu($celltype, $blocklength, $(args...)))
    elseif (package == PKG_METAL)   return :(ParallelStencil.ParallelKernel.trues_metal($celltype, $blocklength, $(args...)))
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.trues_cpu($celltype, $blocklength, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _fill(caller::Module, args...; eltype=nothing, celldims=nothing, celltype=nothing, blocklength=nothing, package::Symbol=get_package(caller))
    celltype    = determine_celltype(caller, eltype, celldims, celltype)
    blocklength = determine_blocklength(blocklength, package)
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.fill_cuda($celltype, $blocklength, $(args...)))
    elseif (package == PKG_AMDGPU)  return :(ParallelStencil.ParallelKernel.fill_amdgpu($celltype, $blocklength, $(args...)))
    elseif (package == PKG_METAL)   return :(ParallelStencil.ParallelKernel.fill_metal($celltype, $blocklength, $(args...)))
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.fill_cpu($celltype, $blocklength, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _fill!(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)    return :(ParallelStencil.ParallelKernel.fill_cuda!($(args...)))
    elseif (package == PKG_AMDGPU)  return :(ParallelStencil.ParallelKernel.fill_amdgpu!($(args...)))
    elseif (package == PKG_METAL)   return :(ParallelStencil.ParallelKernel.fill_metal!($(args...)))
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.fill_cpu!($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _CellType(caller::Module, name; eltype=nothing, fieldnames=nothing, dims=nothing, parametric=nothing)
    if isnothing(fieldnames) @ArgumentError("@CellType: the keyword argument 'fieldnames' is mandatory.") end
    fieldnames = parse_kwargvalues(fieldnames)
    if isnothing(dims)
        dims = [length(fieldnames)]
    else
        dims = parse_kwargvalues(dims)
    end
    if isnothing(parametric) parametric=false end
    if isnothing(eltype)
        eltype = get_numbertype(caller)
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

function determine_celltype(caller::Module, eltype, celldims, celltype)
    if !isnothing(celltype)
        if (!isnothing(celldims) || !isnothing(eltype)) @ArgumentError("the keyword argument 'celltype' is incompatible with the other keyword arguments.") end
    else
        if isnothing(eltype)
            eltype = get_numbertype(caller)
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

 zeros_cpu(::Type{T}, blocklength, args...) where {T<:Number}                      = (check_datatype_cpu(T); Base.zeros(T, args...))                # (blocklength is ignored if neither celldims nor celltype is set)
  ones_cpu(::Type{T}, blocklength, args...) where {T<:Number}                      = (check_datatype_cpu(T); fill_cpu(T, blocklength, 1, args...))  # ...
  rand_cpu(::Type{T}, blocklength, args...) where {T<:Union{Number,Enum}}          = (check_datatype_cpu(T, Bool, Enum); Base.rand(T, args...))     # ...
falses_cpu(::Type{T}, blocklength, args...) where {T<:Bool}                        = Base.falses(args...)                                           # ...
 trues_cpu(::Type{T}, blocklength, args...) where {T<:Bool}                        = Base.trues(args...)                                            # ...  #Note: an alternative would be: fill_cpu(T, true, args...)

 zeros_cpu(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = (check_datatype_cpu(T); fill_cpu(T, blocklength, 0, args...))
  ones_cpu(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = (check_datatype_cpu(T); fill_cpu(T, blocklength, 1, args...))
  rand_cpu(::Type{T}, ::Val{B},    dims)    where {T<:Union{SArray,FieldArray}, B} = (check_datatype_cpu(T, Bool, Enum); blocklen = (B == 0) ? prod(dims) : B; CellArray{T,length(dims),B}(Base.rand(eltype(T), blocklen, prod(size(T)), ceil(Int,prod(dims)/blocklen)), dims))
  rand_cpu(::Type{T}, blocklength, dims...) where {T<:Union{SArray,FieldArray}}    = rand_cpu(T, blocklength, dims)
falses_cpu(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = fill_cpu(T, blocklength, false, args...)
 trues_cpu(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = fill_cpu(T, blocklength, true, args...)

function fill_cpu(::Type{T}, blocklength, x, args...) where {T <: Union{Number, Enum}} # (blocklength is ignored if neither celldims nor celltype is set)
    if (!(eltype(x) <: Number) || (eltype(x) == Bool)) && (eltype(x) != eltype(T)) @ArgumentError("fill: the (element) type of argument 'x' is not a normal number type ($(eltype(x))), but does not match the obtained (default) 'eltype' ($(eltype(T))); automatic conversion to $(eltype(T)) is therefore not attempted. Set the keyword argument 'eltype' accordingly to the element type of 'x' or pass an 'x' of a different (element) type.") end
    check_datatype_cpu(T, Bool, Enum)
    cell = convert(T, x)
    return Base.fill(cell, args...)
end

function fill_cpu(::Type{T}, ::Val{B}, x, args...) where {T <: Union{SArray,FieldArray}, B}
    if (!(eltype(x) <: Number) || (eltype(x) == Bool)) && (eltype(x) != eltype(T)) @ArgumentError("fill: the (element) type of argument 'x' is not a normal number type ($(eltype(x))), but does not match the obtained (default) 'eltype' ($(eltype(T))); automatic conversion to $(eltype(T)) is therefore not attempted. Set the keyword argument 'eltype' accordingly to the element type of 'x' or pass an 'x' of a different (element) type.") end
    check_datatype_cpu(T, Bool, Enum)
    if     (length(x) == 1)         cell = convert(T, fill(convert(eltype(T), x), size(T)))
    elseif (length(x) == length(T)) cell = convert(T, x)
    else                            @ArgumentError("fill: argument 'x' contains the wrong number of elements ($(length(x))). It must be a scalar or contain the number of elements defined by 'celldims'.")
    end
    return CellArrays.fill!(CPUCellArray{T,B}(undef, args...), cell)
end

fill_cpu!(A, x) = Base.fill!(A, construct_cell(A, x))

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

check_datatype_cpu(args...)    = check_datatype(args..., INT_THREADS) # NOTE: if it differs at some point from INT_POLYESTER, then it should be handled differently. For simplicity sake it is kept that way for now.

import Base.length
length(x::Enum) = 1

import Base.eltype
eltype(x::Enum) = typeof(x)
eltype(::Type{T}) where {T <:Enum} = T

import Random
Random.SamplerType{T}() where {T<:Enum} = Random.Sampler(Random.GLOBAL_RNG, instances(T))
