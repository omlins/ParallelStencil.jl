"""
Module FieldAllocators

Provides macros for the allocation of different kind of fields on a grid of size `gridsize`.

# Usage
    using ParallelKernel.FieldAllocators

# Macros

###### Multiple fields at once
- [`@allocate`](@ref)

###### Scalar fields
- [`@Field`](@ref)
- `{X|Y|Z}Field`, e.g. [`@XField`](@ref)
- `B{X|Y|Z}Field`, e.g. [`@BXField`](@ref)
- `{XX|YY|ZZ|XY|XZ|YZ}Field`, e.g. [`@XXField`](@ref)

###### Vector fields
- [`@VectorField`](@ref)
- [`@BVectorField`](@ref)

###### Tensor fields
- [`@TensorField`](@ref)

To see a description of a macro type `?<macroname>` (including the `@`).
"""
module FieldAllocators

using ..Exceptions
import ..ParallelKernel: check_initialized, get_numbertype, extract_kwargvalues, split_args, clean_args, is_same, extract_tuple, extract_kwargs
import ..ParallelKernel: NUMBERTYPE_NONE, FIELDTYPES


##
const ALLOCATE_DOC = """
    @allocate(<keyword arguments>)

Allocate different kinds of fields on a grid of size `gridsize` at once (and initialize them with zeros). Besides convenience and conciseness, this macro ensures that all fields are allocated using the same `gridsize` and is therefore recommended for the allocation of multiple fields.

!!! note "Advanced"
    The `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Mandatory keyword arguments
- `gridsize::Tuple`: the size of the grid.
- `fields::Pair|NTuple{Pair}`: a tuple of pairs (or a single pair) of a field type and a field name or a tuple of field names.

# Keyword arguments
- `allocator`::Macro=@zeros`: the macro to use for the allocation of the field arrays (`@zeros`, `@ones`, `@rand`, `@falses` or `@trues`).
    !!! note "Advanced"
    - `eltype::DataType`: the type of the elements (numbers or indices).

# Examples
    @allocate(gridsize = (nx,ny,nz), 
              fields   = (Field        => (Pt, dτPt, ∇V, Radc, Rog, Mus),
                          VectorField  => (R, dVdτ, dτV),
                          TensorField  => τ,
                          BVectorField => V
              )
    )

See also: [`@allocate`](@ref), [`@Field`](@ref), [`@XField`](@ref), [`@BXField`](@ref), [`@XXField`](@ref), [`@VectorField`](@ref), [`@BVectorField`](@ref), [`@TensorField`](@ref)
"""
@doc ALLOCATE_DOC
macro allocate(args...)
    check_initialized(__module__)
    checkargs_allocate(args...)
    posargs, kwargs_expr = split_args(args)
    gridsize, fields, allocator, eltype = extract_kwargvalues(kwargs_expr, (:gridsize, :fields, :allocator, :eltype), "@allocate")
    esc(_allocate(__module__, posargs...; gridsize=gridsize, fields=fields, allocator=allocator, eltype=eltype))
end


##
const FIELD_DOC = """
    @Field(gridsize)
    @Field(gridsize, allocator)
    @Field(gridsize, allocator, <keyword arguments>)
 
Using the `allocator`, allocate a scalar `Field` on a grid of size `gridsize`.

!!! note "Advanced"
    The `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Arguments
- `gridsize::Tuple`: the size of the grid.
!!! note "Optional argument"
    - `allocator`::Macro=@zeros`: the macro to use for the allocation of the field array (`@zeros`, `@ones`, `@rand`, `@falses` or `@trues`).

# Keyword arguments
- `eltype::DataType`: the type of the elements (numbers or indices).

See also: [`@allocate`](@ref), [`@Field`](@ref), [`@XField`](@ref), [`@BXField`](@ref), [`@XXField`](@ref), [`@VectorField`](@ref), [`@BVectorField`](@ref), [`@TensorField`](@ref)
"""
@doc FIELD_DOC
macro Field(args...)
    check_initialized(__module__)
    checksargs_field_macros(args...)
    posargs, kwargs_expr = split_args(args)
    eltype, = extract_kwargvalues(kwargs_expr, (:eltype,), "@Field")
    posargs = clean_args(posargs)
    esc(_field(__module__, posargs...; eltype=eltype))
end


##
const VECTORFIELD_DOC = """
    @VectorField(gridsize)
    @VectorField(gridsize, allocator)
    @VectorField(gridsize, allocator, <keyword arguments>)

Using the `allocator`, allocate a `VectorField` on a grid of size `gridsize`.

!!! note "Advanced"
    The `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Arguments
- `gridsize::Tuple`: the size of the grid.
!!! note "Optional argument"
    - `allocator`::Macro=@zeros`: the macro to use for the allocation of the field arrays (`@zeros`, `@ones`, `@rand`, `@falses` or `@trues`).

# Keyword arguments
- `eltype::DataType`: the type of the elements (numbers or indices).

See also: [`@allocate`](@ref), [`@Field`](@ref), [`@XField`](@ref), [`@BXField`](@ref), [`@XXField`](@ref), [`@VectorField`](@ref), [`@BVectorField`](@ref), [`@TensorField`](@ref)
"""
@doc VECTORFIELD_DOC
macro VectorField(args...)
    check_initialized(__module__)
    checksargs_field_macros(args...)
    posargs, kwargs_expr = split_args(args)
    eltype, = extract_kwargvalues(kwargs_expr, (:eltype,), "@VectorField")
    posargs = clean_args(posargs)
    esc(_vectorfield(__module__, posargs...; eltype=eltype))
end


##
const BVECTORFIELD_DOC = """
    @BVectorField(gridsize)
    @BVectorField(gridsize, allocator)
    @BVectorField(gridsize, allocator, <keyword arguments>)

Using the `allocator`, allocate a `BVectorField, a vector field including boundaries, on a grid of size `gridsize`.

!!! note "Advanced"
    The `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Arguments
- `gridsize::Tuple`: the size of the grid.
!!! note "Optional argument"
    - `allocator`::Macro=@zeros`: the macro to use for the allocation of the field arrays (`@zeros`, `@ones`, `@rand`, `@falses` or `@trues`).

# Keyword arguments
- `eltype::DataType`: the type of the elements (numbers or indices).

See also: [`@allocate`](@ref), [`@Field`](@ref), [`@XField`](@ref), [`@BXField`](@ref), [`@XXField`](@ref), [`@VectorField`](@ref), [`@BVectorField`](@ref), [`@TensorField`](@ref)
"""
@doc BVECTORFIELD_DOC
macro BVectorField(args...)
    check_initialized(__module__)
    checksargs_field_macros(args...)
    posargs, kwargs_expr = split_args(args)
    eltype, = extract_kwargvalues(kwargs_expr, (:eltype,), "@BVectorField")
    posargs = clean_args(posargs)
    esc(_vectorfield(__module__, posargs...; eltype=eltype, sizetemplate=:B))
end


##
const TENSORFIELD_DOC = """
    @TensorField(gridsize)
    @TensorField(gridsize, allocator)
    @TensorField(gridsize, allocator, <keyword arguments>)

Using the `allocator`, allocate a `TensorField` on a grid of size `gridsize`.

!!! note "Advanced"
    The `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Arguments
- `gridsize::Tuple`: the size of the grid.
!!! note "Optional argument"
    - `allocator`::Macro=@zeros`: the macro to use for the allocation of the field arrays (`@zeros`, `@ones`, `@rand`, `@falses` or `@trues`).

# Keyword arguments
- `eltype::DataType`: the type of the elements (numbers or indices).

See also: [`@allocate`](@ref), [`@Field`](@ref), [`@XField`](@ref), [`@BXField`](@ref), [`@XXField`](@ref), [`@VectorField`](@ref), [`@BVectorField`](@ref), [`@TensorField`](@ref)
"""
@doc TENSORFIELD_DOC
macro TensorField(args...)
    check_initialized(__module__)
    checksargs_field_macros(args...)
    posargs, kwargs_expr = split_args(args)
    eltype, = extract_kwargvalues(kwargs_expr, (:eltype,), "@TensorField")
    posargs = clean_args(posargs)
    esc(_tensorfield(__module__, posargs...; eltype=eltype))
end


##
const VECTORFIELD_COMP_DOC = """
    @{X|Y|Z}Field(gridsize)
    @{X|Y|Z}Field(gridsize, allocator)
    @{X|Y|Z}Field(gridsize, allocator, <keyword arguments>)

Using the `allocator`, allocate a `{X|Y|Z}Field`, a scalar field of the same size as the {X|Y|Z}-component of a `VectorField`, on a grid of size `gridsize`.

!!! note "Advanced"
    The `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Arguments
- `gridsize::Tuple`: the size of the grid.
!!! note "Optional argument"
    - `allocator`::Macro=@zeros`: the macro to use for the allocation of the field array (`@zeros`, `@ones`, `@rand`, `@falses` or `@trues`).

# Keyword arguments
- `eltype::DataType`: the type of the elements (numbers or indices).

See also: [`@allocate`](@ref), [`@Field`](@ref), [`@XField`](@ref), [`@BXField`](@ref), [`@XXField`](@ref), [`@VectorField`](@ref), [`@BVectorField`](@ref), [`@TensorField`](@ref)
"""

@doc VECTORFIELD_COMP_DOC
macro XField(args...)
    check_initialized(__module__)
    checksargs_field_macros(args...)
    posargs, kwargs_expr = split_args(args)
    eltype, = extract_kwargvalues(kwargs_expr, (:eltype,), "@XField")
    posargs = clean_args(posargs)
    esc(_field(__module__, posargs...; eltype=eltype, sizetemplate=:X))
end

@doc VECTORFIELD_COMP_DOC
macro YField(args...)
    check_initialized(__module__)
    checksargs_field_macros(args...)
    posargs, kwargs_expr = split_args(args)
    eltype, = extract_kwargvalues(kwargs_expr, (:eltype,), "@YField")
    posargs = clean_args(posargs)
    esc(_field(__module__, posargs...; eltype=eltype, sizetemplate=:Y))
end

@doc VECTORFIELD_COMP_DOC
macro ZField(args...)
    check_initialized(__module__)
    checksargs_field_macros(args...)
    posargs, kwargs_expr = split_args(args)
    eltype, = extract_kwargvalues(kwargs_expr, (:eltype,), "@ZField")
    posargs = clean_args(posargs)
    esc(_field(__module__, posargs...; eltype=eltype, sizetemplate=:Z))
end


##
const BVECTORFIELD_COMP_DOC = """
    @B{X|Y|Z}Field(gridsize)
    @B{X|Y|Z}Field(gridsize, allocator)
    @B{X|Y|Z}Field(gridsize, allocator, <keyword arguments>)

Using the `allocator`, allocate a `B{X|Y|Z}Field`, a scalar field of the same size as the {X|Y|Z}-component of a `BVectorField` (a vector field including boundaries), on a grid of size `gridsize`.

!!! note "Advanced"
    The `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Arguments
- `gridsize::Tuple`: the size of the grid.
!!! note "Optional argument"
    - `allocator`::Macro=@zeros`: the macro to use for the allocation of the field array (`@zeros`, `@ones`, `@rand`, `@falses` or `@trues`).

# Keyword arguments
- `eltype::DataType`: the type of the elements (numbers or indices).

See also: [`@allocate`](@ref), [`@Field`](@ref), [`@XField`](@ref), [`@BXField`](@ref), [`@XXField`](@ref), [`@VectorField`](@ref), [`@BVectorField`](@ref), [`@TensorField`](@ref)
"""

@doc BVECTORFIELD_COMP_DOC
macro BXField(args...)
    check_initialized(__module__)
    checksargs_field_macros(args...)
    posargs, kwargs_expr = split_args(args)
    eltype, = extract_kwargvalues(kwargs_expr, (:eltype,), "@BXField")
    posargs = clean_args(posargs)
    esc(_field(__module__, posargs...; eltype=eltype, sizetemplate=:BX))
end

@doc BVECTORFIELD_COMP_DOC
macro BYField(args...)
    check_initialized(__module__)
    checksargs_field_macros(args...)
    posargs, kwargs_expr = split_args(args)
    eltype, = extract_kwargvalues(kwargs_expr, (:eltype,), "@BYField")
    posargs = clean_args(posargs)
    esc(_field(__module__, posargs...; eltype=eltype, sizetemplate=:BY))
end

@doc BVECTORFIELD_COMP_DOC
macro BZField(args...)
    check_initialized(__module__)
    checksargs_field_macros(args...)
    posargs, kwargs_expr = split_args(args)
    eltype, = extract_kwargvalues(kwargs_expr, (:eltype,), "@BZField")
    posargs = clean_args(posargs)
    esc(_field(__module__, posargs...; eltype=eltype, sizetemplate=:BZ))
end


##
const TENSORFIELD_COMP_DOC = """
    @{XX|YY|ZZ|XY|XZ|YZ}Field(gridsize)
    @{XX|YY|ZZ|XY|XZ|YZ}Field(gridsize, allocator)
    @{XX|YY|ZZ|XY|XZ|YZ}Field(gridsize, allocator, <keyword arguments>)

Using the `allocator`, allocate a `{XX|YY|ZZ|XY|XZ|YZ}Field`, a scalar field of the same size as the {XX|YY|ZZ|XY|XZ|YZ}-component of a `TensorField`, on a grid of size `gridsize`.

!!! note "Advanced"
    The `eltype` can be explicitly passed as keyword argument in order to be used instead of the default `numbertype` chosen with [`@init_parallel_kernel`](@ref). If no default `numbertype` was chosen [`@init_parallel_kernel`](@ref), then the keyword argument `eltype` is mandatory. This needs to be used with care to ensure that no datatype conversions occur in performance critical computations.

# Arguments
- `gridsize::Tuple`: the size of the grid.
!!! note "Optional argument"
    - `allocator`::Macro=@zeros`: the macro to use for the allocation of the field array (`@zeros`, `@ones`, `@rand`, `@falses` or `@trues`).

# Keyword arguments
- `eltype::DataType`: the type of the elements (numbers or indices).

See also: [`@allocate`](@ref), [`@Field`](@ref), [`@XField`](@ref), [`@BXField`](@ref), [`@XXField`](@ref), [`@VectorField`](@ref), [`@BVectorField`](@ref), [`@TensorField`](@ref)
"""

@doc TENSORFIELD_COMP_DOC
macro XXField(args...)
    check_initialized(__module__)
    checksargs_field_macros(args...)
    posargs, kwargs_expr = split_args(args)
    eltype, = extract_kwargvalues(kwargs_expr, (:eltype,), "@XXField")
    posargs = clean_args(posargs)
    esc(_field(__module__, posargs...; eltype=eltype, sizetemplate=:XX))
end

@doc TENSORFIELD_COMP_DOC
macro YYField(args...)
    check_initialized(__module__)
    checksargs_field_macros(args...)
    posargs, kwargs_expr = split_args(args)
    eltype, = extract_kwargvalues(kwargs_expr, (:eltype,), "@YYField")
    posargs = clean_args(posargs)
    esc(_field(__module__, posargs...; eltype=eltype, sizetemplate=:YY))
end

@doc TENSORFIELD_COMP_DOC
macro ZZField(args...)
    check_initialized(__module__)
    checksargs_field_macros(args...)
    posargs, kwargs_expr = split_args(args)
    eltype, = extract_kwargvalues(kwargs_expr, (:eltype,), "@ZZField")
    posargs = clean_args(posargs)
    esc(_field(__module__, posargs...; eltype=eltype, sizetemplate=:ZZ))
end

@doc TENSORFIELD_COMP_DOC
macro XYField(args...)
    check_initialized(__module__)
    checksargs_field_macros(args...)
    posargs, kwargs_expr = split_args(args)
    eltype, = extract_kwargvalues(kwargs_expr, (:eltype,), "@XYField")
    posargs = clean_args(posargs)
    esc(_field(__module__, posargs...; eltype=eltype, sizetemplate=:XY))
end

@doc TENSORFIELD_COMP_DOC
macro XZField(args...)
    check_initialized(__module__)
    checksargs_field_macros(args...)
    posargs, kwargs_expr = split_args(args)
    eltype, = extract_kwargvalues(kwargs_expr, (:eltype,), "@XZField")
    posargs = clean_args(posargs)
    esc(_field(__module__, posargs...; eltype=eltype, sizetemplate=:XZ))
end

@doc TENSORFIELD_COMP_DOC
macro YZField(args...)
    check_initialized(__module__)
    checksargs_field_macros(args...)
    posargs, kwargs_expr = split_args(args)
    eltype, = extract_kwargvalues(kwargs_expr, (:eltype,), "@YZField")
    posargs = clean_args(posargs)
    esc(_field(__module__, posargs...; eltype=eltype, sizetemplate=:YZ))
end


## ARGUMENT CHECKS

function checkargs_allocate(args...)
    if isempty(args) @ArgumentError("arguments missing.") end
    posargs, kwargs_expr = split_args(args)
    if length(posargs) > 0 @ArgumentError("no positional arguments are allowed.") end
    if length(kwargs_expr) < 2 @ArgumentError("the gridsize and the fields keyword argument are mandatory.") end
    if length(kwargs_expr) > 4 @ArgumentError("too many keyword arguments.") end
end

function checksargs_field_macros(args...)
    if isempty(args) @ArgumentError("arguments missing.") end
    posargs, kwargs_expr = split_args(args)
    posargs = clean_args(posargs)
    if isempty(posargs) @ArgumentError("the gridsize positional argument is mandatory.") end
    if length(posargs) > 2 @ArgumentError("too many positional arguments.") end
    if (length(posargs) == 2) && !(any(is_same.((posargs[2],), (:@zeros, :@ones, :@rand, :@falses, :@trues)))) @ArgumentError("the second positional argument must be a field allocator macro.") end
    if length(kwargs_expr) > 1 @ArgumentError("the only allowed keyword argument is eltype.") end
end


## ALLOCATOR FUNCTIONS

function _allocate(caller::Module; gridsize=nothing, fields=nothing, allocator=nothing, eltype=nothing)
    eltype = determine_eltype(caller, eltype)
    allocator = isnothing(allocator) ? (:@zeros) : allocator # NOTE: this cannot be set in signature because it can receive the value `nothing`.
    if isnothing(gridsize) || isnothing(fields) @ModuleInternalError("gridsize and fields are mandatory.") end
    fields_expr = extract_tuple(fields; nested=true)
    fields_kwargs = pairs(extract_kwargs(caller, fields_expr, FIELDTYPES, "@allocate"; separator=:(=>)))
    allocations = []
    for (T, As_expr) in fields_kwargs
        As = extract_tuple(As_expr)
        for A in As
            if     (T == :Field)        allocation = :($A = @Field($gridsize, $allocator, eltype=$eltype))
            elseif (T == :XField)       allocation = :($A = @XField($gridsize, $allocator, eltype=$eltype))
            elseif (T == :YField)       allocation = :($A = @YField($gridsize, $allocator, eltype=$eltype))
            elseif (T == :ZField)       allocation = :($A = @ZField($gridsize, $allocator, eltype=$eltype))
            elseif (T == :BXField)      allocation = :($A = @BXField($gridsize, $allocator, eltype=$eltype))
            elseif (T == :BYField)      allocation = :($A = @BYField($gridsize, $allocator, eltype=$eltype))
            elseif (T == :BZField)      allocation = :($A = @BZField($gridsize, $allocator, eltype=$eltype))
            elseif (T == :XXField)      allocation = :($A = @XXField($gridsize, $allocator, eltype=$eltype))
            elseif (T == :YYField)      allocation = :($A = @YYField($gridsize, $allocator, eltype=$eltype))
            elseif (T == :ZZField)      allocation = :($A = @ZZField($gridsize, $allocator, eltype=$eltype))
            elseif (T == :XYField)      allocation = :($A = @XYField($gridsize, $allocator, eltype=$eltype))
            elseif (T == :XZField)      allocation = :($A = @XZField($gridsize, $allocator, eltype=$eltype))
            elseif (T == :YZField)      allocation = :($A = @YZField($gridsize, $allocator, eltype=$eltype))
            elseif (T == :VectorField)  allocation = :($A = @VectorField($gridsize, $allocator, eltype=$eltype))
            elseif (T == :BVectorField) allocation = :($A = @BVectorField($gridsize, $allocator, eltype=$eltype))
            elseif (T == :TensorField)  allocation = :($A = @TensorField($gridsize, $allocator, eltype=$eltype))
            else @ModuleInternalError("unexpected field type.")
            end
            push!(allocations, allocation)
        end
    end
    return quote $(allocations...) end
end

function _field(caller::Module, gridsize, allocator=:@zeros; eltype=nothing, sizetemplate=nothing)
    eltype   = determine_eltype(caller, eltype)
    if     (sizetemplate == :X)   arraysize = :($gridsize .+ ((length($gridsize)==3) ? (-1,-2,-2) : (length($gridsize)==2) ? (-1,-2) : -1))
    elseif (sizetemplate == :Y)   arraysize = :($gridsize .+ ((length($gridsize)==3) ? (-2,-1,-2) : (length($gridsize)==2) ? (-2,-1) : -2))
    elseif (sizetemplate == :Z)   arraysize = :($gridsize .+ ((length($gridsize)==3) ? (-2,-2,-1) : (length($gridsize)==2) ? (-2,-2) : -2))
    elseif (sizetemplate == :BX)  arraysize = :($gridsize .+ ((length($gridsize)==3) ? (+1, 0, 0) : (length($gridsize)==2) ? (+1, 0) : +1))
    elseif (sizetemplate == :BY)  arraysize = :($gridsize .+ ((length($gridsize)==3) ? ( 0,+1, 0) : (length($gridsize)==2) ? ( 0,+1) :  0))
    elseif (sizetemplate == :BZ)  arraysize = :($gridsize .+ ((length($gridsize)==3) ? ( 0, 0,+1) : (length($gridsize)==2) ? ( 0, 0) :  0))
    elseif (sizetemplate == :XX)  arraysize = :($gridsize .+ ((length($gridsize)==3) ? ( 0,-2,-2) : (length($gridsize)==2) ? ( 0,-2) :  0))
    elseif (sizetemplate == :YY)  arraysize = :($gridsize .+ ((length($gridsize)==3) ? (-2, 0,-2) : (length($gridsize)==2) ? (-2, 0) : -2))
    elseif (sizetemplate == :ZZ)  arraysize = :($gridsize .+ ((length($gridsize)==3) ? (-2,-2, 0) : (length($gridsize)==2) ? (-2,-2) : -2))
    elseif (sizetemplate == :XY)  arraysize = :($gridsize .+ ((length($gridsize)==3) ? (-1,-1,-2) : (length($gridsize)==2) ? (-1,-1) : -1))
    elseif (sizetemplate == :XZ)  arraysize = :($gridsize .+ ((length($gridsize)==3) ? (-1,-2,-1) : (length($gridsize)==2) ? (-1,-2) : -1))
    elseif (sizetemplate == :YZ)  arraysize = :($gridsize .+ ((length($gridsize)==3) ? (-2,-1,-1) : (length($gridsize)==2) ? (-2,-1) : -2))
    else                          arraysize = gridsize
    end
    if     is_same(allocator, :@zeros)  return :(ParallelStencil.ParallelKernel.@zeros($arraysize..., eltype=$eltype))
    elseif is_same(allocator, :@ones)   return :(ParallelStencil.ParallelKernel.@ones($arraysize..., eltype=$eltype))
    elseif is_same(allocator, :@rand)   return :(ParallelStencil.ParallelKernel.@rand($arraysize..., eltype=$eltype))
    elseif is_same(allocator, :@falses) return :(ParallelStencil.ParallelKernel.@falses($arraysize..., eltype=$eltype))
    elseif is_same(allocator, :@trues)  return :(ParallelStencil.ParallelKernel.@trues($arraysize..., eltype=$eltype))
    else @ModuleInternalError("unexpected allocator macro.")
    end
end

function _vectorfield(caller::Module, gridsize, allocator=:@zeros; eltype=nothing, sizetemplate=nothing)
    eltype = determine_eltype(caller, eltype)
    if (sizetemplate == :B)
        return :((length($gridsize)==3) ? (x = ParallelStencil.ParallelKernel.FieldAllocators.@BXField($gridsize, $allocator, eltype=$eltype),
                                           y = ParallelStencil.ParallelKernel.FieldAllocators.@BYField($gridsize, $allocator, eltype=$eltype),
                                           z = ParallelStencil.ParallelKernel.FieldAllocators.@BZField($gridsize, $allocator, eltype=$eltype)) :
                  length($gridsize)==2  ? (x = ParallelStencil.ParallelKernel.FieldAllocators.@BXField($gridsize, $allocator, eltype=$eltype),
                                           y = ParallelStencil.ParallelKernel.FieldAllocators.@BYField($gridsize, $allocator, eltype=$eltype)) :
                                          (x = ParallelStencil.ParallelKernel.FieldAllocators.@BXField($gridsize, $allocator, eltype=$eltype),))
    else
        return :((length($gridsize)==3) ? (x = ParallelStencil.ParallelKernel.FieldAllocators.@XField($gridsize, $allocator, eltype=$eltype),
                                           y = ParallelStencil.ParallelKernel.FieldAllocators.@YField($gridsize, $allocator, eltype=$eltype),
                                           z = ParallelStencil.ParallelKernel.FieldAllocators.@ZField($gridsize, $allocator, eltype=$eltype)) :
                  length($gridsize)==2  ? (x = ParallelStencil.ParallelKernel.FieldAllocators.@XField($gridsize, $allocator, eltype=$eltype),
                                           y = ParallelStencil.ParallelKernel.FieldAllocators.@YField($gridsize, $allocator, eltype=$eltype)) :
                                          (x = ParallelStencil.ParallelKernel.FieldAllocators.@XField($gridsize, $allocator, eltype=$eltype),))
    end    
end

function _tensorfield(caller::Module, gridsize, allocator=:@zeros; eltype=nothing)
    eltype = determine_eltype(caller, eltype)
    return :((length($gridsize)==3) ? (xx = ParallelStencil.ParallelKernel.FieldAllocators.@XXField($gridsize, $allocator, eltype=$eltype),
                                       yy = ParallelStencil.ParallelKernel.FieldAllocators.@YYField($gridsize, $allocator, eltype=$eltype),
                                       zz = ParallelStencil.ParallelKernel.FieldAllocators.@ZZField($gridsize, $allocator, eltype=$eltype),
                                       xy = ParallelStencil.ParallelKernel.FieldAllocators.@XYField($gridsize, $allocator, eltype=$eltype),
                                       xz = ParallelStencil.ParallelKernel.FieldAllocators.@XZField($gridsize, $allocator, eltype=$eltype),
                                       yz = ParallelStencil.ParallelKernel.FieldAllocators.@YZField($gridsize, $allocator, eltype=$eltype)) :
              length($gridsize)==2  ? (xx = ParallelStencil.ParallelKernel.FieldAllocators.@XXField($gridsize, $allocator, eltype=$eltype),
                                       yy = ParallelStencil.ParallelKernel.FieldAllocators.@YYField($gridsize, $allocator, eltype=$eltype),
                                       xy = ParallelStencil.ParallelKernel.FieldAllocators.@XYField($gridsize, $allocator, eltype=$eltype)) :
                                      (xx = ParallelStencil.ParallelKernel.FieldAllocators.@XXField($gridsize, $allocator, eltype=$eltype),))
end

function determine_eltype(caller::Module, eltype)
    if isnothing(eltype)
        eltype = get_numbertype(caller)
        if (eltype == NUMBERTYPE_NONE) @ArgumentError("the keyword argument 'eltype' is mandatory in @allocate, @Field, @VectorField, @TensorField, @XField, @YField, @ZField, @XXField, @YYField, @ZZField, @XYField, @XZField and @YZField when no default is set.") end
    end
    return eltype
end


## Exports

export @allocate, @Field, @VectorField, @BVectorField, @TensorField, @XField, @BXField, @YField, @BYField, @ZField, @BZField, @XXField, @YYField, @ZZField, @XYField, @XZField, @YZField


end # Module FieldAllocators
