"""
Module FieldAllocators

Provides macros for the allocation of different kind of fields on a grid of size `gridsize`.

# Usage
    using ParallelStencil.FieldAllocators

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
    import ..ParallelKernel
    import ..ParallelStencil: check_initialized
    @doc replace(ParallelKernel.FieldAllocators.ALLOCATE_DOC,          "@init_parallel_kernel" => "@init_parallel_stencil") macro allocate(args...)     check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@allocate($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.FIELD_DOC,             "@init_parallel_kernel" => "@init_parallel_stencil") macro Field(args...)        check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@Field($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.VECTORFIELD_DOC,       "@init_parallel_kernel" => "@init_parallel_stencil") macro VectorField(args...)  check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@VectorField($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.BVECTORFIELD_DOC,      "@init_parallel_kernel" => "@init_parallel_stencil") macro BVectorField(args...) check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@BVectorField($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.TENSORFIELD_DOC,       "@init_parallel_kernel" => "@init_parallel_stencil") macro TensorField(args...)  check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@TensorField($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.VECTORFIELD_COMP_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro XField(args...)       check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XField($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.BVECTORFIELD_COMP_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro BXField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@BXField($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.VECTORFIELD_COMP_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro YField(args...)       check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@YField($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.BVECTORFIELD_COMP_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro BYField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@BYField($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.VECTORFIELD_COMP_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro ZField(args...)       check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@ZField($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.BVECTORFIELD_COMP_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro BZField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@BZField($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.TENSORFIELD_COMP_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro XXField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XXField($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.TENSORFIELD_COMP_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro YYField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@YYField($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.TENSORFIELD_COMP_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro ZZField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@ZZField($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.TENSORFIELD_COMP_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro XYField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XYField($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.TENSORFIELD_COMP_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro XZField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XZField($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.TENSORFIELD_COMP_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro YZField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@YZField($(args...)))); end

    export @allocate, @Field, @VectorField, @BVectorField, @TensorField, @XField, @BXField, @YField, @BYField, @ZField, @BZField, @XXField, @YYField, @ZZField, @XYField, @XZField, @YZField
end