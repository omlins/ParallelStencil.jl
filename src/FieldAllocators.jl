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

    macro IField(args...)        check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@IField($(args...)))); end
    macro XXYField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XXYField($(args...)))); end
    macro XYYField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XYYField($(args...)))); end
    macro XYZField(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XYZField($(args...)))); end
    macro XXYZField(args...)     check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XXYZField($(args...)))); end
    macro XYYZField(args...)     check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XYYZField($(args...)))); end
    macro XYZZField(args...)     check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XYZZField($(args...)))); end
    macro XXYYField(args...)     check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XXYYField($(args...)))); end
    macro XXZZField(args...)     check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XXZZField($(args...)))); end
    macro YYZZField(args...)     check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@YYZZField($(args...)))); end
    macro XXYYZField(args...)    check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XXYYZField($(args...)))); end
    macro XYYZZField(args...)    check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XYYZZField($(args...)))); end
    macro XXYZZField(args...)    check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XXYZZField($(args...)))); end

    @doc replace(ParallelKernel.FieldAllocators.FIELD2B_DOC,        "@init_parallel_kernel" => "@init_parallel_stencil") macro Field2B(args...)        check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@Field2B($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.VECTORFIELD2B_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro VectorField2B(args...)  check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@VectorField2B($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.BVECTORFIELD2B_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro BVectorField2B(args...) check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@BVectorField2B($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.TENSORFIELD2B_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil") macro TensorField2B(args...)  check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@TensorField2B($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.FIELD2B_COMP_DOC,   "@init_parallel_kernel" => "@init_parallel_stencil") macro XField2B(args...)       check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XField2B($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.FIELD2B_COMP_DOC,   "@init_parallel_kernel" => "@init_parallel_stencil") macro BXField2B(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@BXField2B($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.FIELD2B_COMP_DOC,   "@init_parallel_kernel" => "@init_parallel_stencil") macro YField2B(args...)       check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@YField2B($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.FIELD2B_COMP_DOC,   "@init_parallel_kernel" => "@init_parallel_stencil") macro BYField2B(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@BYField2B($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.FIELD2B_COMP_DOC,   "@init_parallel_kernel" => "@init_parallel_stencil") macro ZField2B(args...)       check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@ZField2B($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.FIELD2B_COMP_DOC,   "@init_parallel_kernel" => "@init_parallel_stencil") macro BZField2B(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@BZField2B($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.FIELD2B_COMP_DOC,   "@init_parallel_kernel" => "@init_parallel_stencil") macro XXField2B(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XXField2B($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.FIELD2B_COMP_DOC,   "@init_parallel_kernel" => "@init_parallel_stencil") macro YYField2B(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@YYField2B($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.FIELD2B_COMP_DOC,   "@init_parallel_kernel" => "@init_parallel_stencil") macro ZZField2B(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@ZZField2B($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.FIELD2B_COMP_DOC,   "@init_parallel_kernel" => "@init_parallel_stencil") macro XYField2B(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XYField2B($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.FIELD2B_COMP_DOC,   "@init_parallel_kernel" => "@init_parallel_stencil") macro XZField2B(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@XZField2B($(args...)))); end
    @doc replace(ParallelKernel.FieldAllocators.FIELD2B_COMP_DOC,   "@init_parallel_kernel" => "@init_parallel_stencil") macro YZField2B(args...)      check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.FieldAllocators.@YZField2B($(args...)))); end

    export @allocate, @Field, @VectorField, @BVectorField, @TensorField, @XField, @BXField, @YField, @BYField, @ZField, @BZField, @XXField, @YYField, @ZZField, @XYField, @XZField, @YZField, @Field2B, @VectorField2B, @BVectorField2B, @TensorField2B, @XField2B, @BXField2B, @YField2B, @BYField2B, @ZField2B, @BZField2B, @XXField2B, @YYField2B, @ZZField2B, @XYField2B, @XZField2B, @YZField2B
end