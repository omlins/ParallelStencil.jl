"""
Module Fields

Provides macros for the allocation of different kind of fields on a grid of size `gridsize`.

# Usage
    using ParallelStencil.Fields

# Macros

###### Multiple fields at once
- [`@allocate`](@ref)

###### Scalar fields
- [`@Field`](@ref)
- `{X|Y|Z}Fields`, e.g. [`@XField`](@ref)
- `B{X|Y|Z}Fields`, e.g. [`@BXField`](@ref)
- `{XX|YY|ZZ|XY|XZ|YZ}Fields`, e.g. [`@XXField`](@ref)

###### Vector fields
- [`@VectorField`](@ref)
- [`@BVectorField`](@ref)

###### Tensor fields
- [`@TensorField`](@ref)

To see a description of a macro type `?<macroname>` (including the `@`).
"""
module Fields
    import ..ParallelKernel
    @doc replace(ParallelKernel.Fields.ALLOCATE_DOC,          "@init_parallel_kernel" => "@init_parallel_stencil") macro allocate(args...)     check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.Fields.@allocate($(args...)))); end
    @doc replace(ParallelKernel.Fields.FIELD_DOC,             "@init_parallel_kernel" => "@init_parallel_stencil") macro Field(args...)        check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.Fields.@Field($(args...)))); end
    @doc replace(ParallelKernel.Fields.VECTORFIELD_DOC,       "@init_parallel_kernel" => "@init_parallel_stencil") macro VectorField(args...)  check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.Fields.@VectorField($(args...)))); end
    @doc replace(ParallelKernel.Fields.BVECTORFIELD_DOC,      "@init_parallel_kernel" => "@init_parallel_stencil") macro BVectorField(args...) check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.Fields.@BVectorField($(args...)))); end
    @doc replace(ParallelKernel.Fields.TENSORFIELD_DOC,       "@init_parallel_kernel" => "@init_parallel_stencil") macro TensorField(args...)  check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.Fields.@TensorField($(args...)))); end
    @doc replace(ParallelKernel.Fields.VECTORFIELD_COMP_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro XField(args...)       check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.Fields.@XField($(args...)))); end
    @doc replace(ParallelKernel.Fields.BVECTORFIELD_COMP_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro BXField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.Fields.@BXField($(args...)))); end
    @doc replace(ParallelKernel.Fields.VECTORFIELD_COMP_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro YField(args...)       check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.Fields.@YField($(args...)))); end
    @doc replace(ParallelKernel.Fields.BVECTORFIELD_COMP_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro BYField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.Fields.@BYField($(args...)))); end
    @doc replace(ParallelKernel.Fields.VECTORFIELD_COMP_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro ZField(args...)       check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.Fields.@ZField($(args...)))); end
    @doc replace(ParallelKernel.Fields.BVECTORFIELD_COMP_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro BZField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.Fields.@BZField($(args...)))); end
    @doc replace(ParallelKernel.Fields.TENSORFIELD_COMP_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro XXField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.Fields.@XXField($(args...)))); end
    @doc replace(ParallelKernel.Fields.TENSORFIELD_COMP_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro YYField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.Fields.@YYField($(args...)))); end
    @doc replace(ParallelKernel.Fields.TENSORFIELD_COMP_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro ZZField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.Fields.@ZZField($(args...)))); end
    @doc replace(ParallelKernel.Fields.TENSORFIELD_COMP_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro XYField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.Fields.@XYField($(args...)))); end
    @doc replace(ParallelKernel.Fields.TENSORFIELD_COMP_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro XZField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.Fields.@XZField($(args...)))); end
    @doc replace(ParallelKernel.Fields.TENSORFIELD_COMP_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro YZField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.Fields.@YZField($(args...)))); end

    export @allocate, @Field, @VectorField, @BVectorField, @TensorField, @XField, @BXField, @YField, @BYField, @ZField, @BZField, @XXField, @YYField, @ZZField, @XYField, @XZField, @YZField
end