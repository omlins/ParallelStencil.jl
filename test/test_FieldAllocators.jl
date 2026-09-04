using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_THREADS, PKG_POLYESTER, PKG_KERNELABSTRACTIONS, FIELDTYPES
import ParallelStencil: @require
using ParallelStencil.Exceptions
using ParallelStencil.FieldAllocators
TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end
@static if PKG_AMDGPU in TEST_PACKAGES
    import AMDGPU
    if !AMDGPU.functional() TEST_PACKAGES = filter!(x->x≠PKG_AMDGPU, TEST_PACKAGES) end
end
@static if PKG_METAL in TEST_PACKAGES
    import Metal
    if !Metal.functional() TEST_PACKAGES = filter!(x->x≠PKG_METAL, TEST_PACKAGES) end
end
@static if PKG_KERNELABSTRACTIONS in TEST_PACKAGES
    import KernelAbstractions
    if !KernelAbstractions.functional(KernelAbstractions.CPU()) TEST_PACKAGES = filter!(x->x≠PKG_KERNELABSTRACTIONS, TEST_PACKAGES) end
end
@static if PKG_POLYESTER in TEST_PACKAGES
    import Polyester
end
Base.retry_load_extensions() # Potentially needed to load the extensions after the packages have been filtered.


@static for package in TEST_PACKAGES
    FloatDefault = (package == PKG_METAL) ? Float32 : Float64 # Metal does not support Float64

eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. 2B field allocator macros" begin
            @require !@is_initialized()
            @init_parallel_stencil($package, $FloatDefault, 3, nonconst_metadata=true)
            @require @is_initialized()

            nxyz = (8, 8, 8)

            @testset "@Field2B" begin
                result = @Field2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out) == nxyz
                @test typeof(result.in) == typeof(@Field(nxyz))
            end;

            @testset "@XField2B" begin
                result = @XField2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out)
                @test typeof(result.in) == typeof(@XField(nxyz))
            end;

            @testset "@YField2B" begin
                result = @YField2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out)
                @test typeof(result.in) == typeof(@YField(nxyz))
            end;

            @testset "@ZField2B" begin
                result = @ZField2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out)
                @test typeof(result.in) == typeof(@ZField(nxyz))
            end;

            @testset "@BXField2B" begin
                result = @BXField2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out)
                @test typeof(result.in) == typeof(@BXField(nxyz))
            end;

            @testset "@BYField2B" begin
                result = @BYField2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out)
                @test typeof(result.in) == typeof(@BYField(nxyz))
            end;

            @testset "@BZField2B" begin
                result = @BZField2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out)
                @test typeof(result.in) == typeof(@BZField(nxyz))
            end;

            @testset "@XXField2B" begin
                result = @XXField2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out)
                @test typeof(result.in) == typeof(@XXField(nxyz))
            end;

            @testset "@YYField2B" begin
                result = @YYField2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out)
                @test typeof(result.in) == typeof(@YYField(nxyz))
            end;

            @testset "@ZZField2B" begin
                result = @ZZField2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out)
                @test typeof(result.in) == typeof(@ZZField(nxyz))
            end;

            @testset "@XYField2B" begin
                result = @XYField2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out)
                @test typeof(result.in) == typeof(@XYField(nxyz))
            end;

            @testset "@XZField2B" begin
                result = @XZField2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out)
                @test typeof(result.in) == typeof(@XZField(nxyz))
            end;

            @testset "@YZField2B" begin
                result = @YZField2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out)
                @test typeof(result.in) == typeof(@YZField(nxyz))
            end;

            @testset "@VectorField2B" begin
                result = @VectorField2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test keys(result.in) == (:x, :y, :z)
                @test keys(result.out) == (:x, :y, :z)
                @test size(result.in.x) == size(result.out.x) == size(@VectorField(nxyz).x)
                @test size(result.in.y) == size(result.out.y) == size(@VectorField(nxyz).y)
                @test size(result.in.z) == size(result.out.z) == size(@VectorField(nxyz).z)
            end;

            @testset "@BVectorField2B" begin
                result = @BVectorField2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test keys(result.in) == (:x, :y, :z)
                @test keys(result.out) == (:x, :y, :z)
                @test size(result.in.x) == size(result.out.x) == size(@BVectorField(nxyz).x)
                @test size(result.in.y) == size(result.out.y) == size(@BVectorField(nxyz).y)
                @test size(result.in.z) == size(result.out.z) == size(@BVectorField(nxyz).z)
            end;

            @testset "@TensorField2B" begin
                result = @TensorField2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test keys(result.in) == (:xx, :yy, :zz, :xy, :xz, :yz)
                @test keys(result.out) == (:xx, :yy, :zz, :xy, :xz, :yz)
                @test size(result.in.xx) == size(result.out.xx) == size(@TensorField(nxyz).xx)
                @test size(result.in.yy) == size(result.out.yy) == size(@TensorField(nxyz).yy)
                @test size(result.in.zz) == size(result.out.zz) == size(@TensorField(nxyz).zz)
            end;

            @reset_parallel_stencil()
        end;

        @testset "3. @allocate with 2B kinds" begin
            @require !@is_initialized()
            @init_parallel_stencil($package, $FloatDefault, 3, nonconst_metadata=true)
            @require @is_initialized()

            nxyz = (8, 8, 8)

            @testset "Field2B, BVectorField2B, Field" begin
                @allocate(gridsize=nxyz, fields=(Field2B => (A, B), BVectorField2B => (V,), Field => (C,)))
                @test keys(A) == (:in, :out)
                @test keys(B) == (:in, :out)
                @test keys(V) == (:in, :out)
                @test keys(V.in) == (:x, :y, :z)
                @test size(C) == nxyz
                @test size(A.in) == size(A.out) == nxyz
                @test typeof(A.in) == typeof(@Field(nxyz))
                @test size(V.in.x) == size(V.out.x) == size(@BVectorField(nxyz).x)
                @test size(V.in.y) == size(V.out.y) == size(@BVectorField(nxyz).y)
                @test size(V.in.z) == size(V.out.z) == size(@BVectorField(nxyz).z)
            end;

            @testset "TensorField2B and XField2B" begin
                @allocate(gridsize=nxyz, fields=(TensorField2B => (T,), XField2B => (XF,)))
                @test keys(T) == (:in, :out)
                @test keys(T.in) == (:xx, :yy, :zz, :xy, :xz, :yz)
                @test size(T.in.xx) == size(T.out.xx) == size(@TensorField(nxyz).xx)
                @test keys(XF) == (:in, :out)
                @test size(XF.in) == size(XF.out) == size(@XField(nxyz))
                @test typeof(XF.in) == typeof(@XField(nxyz))
            end;

            @testset "allocator=@ones" begin
                @allocate(gridsize=nxyz, fields=(Field2B => (A,), Field => (C,)), allocator=@ones)
                @test keys(A) == (:in, :out)
                @test all(A.in .== 1.0)
                @test all(A.out .== 1.0)
                @test all(C .== 1.0)
            end;

            @testset "allocator=@rand" begin
                @allocate(gridsize=nxyz, fields=(Field2B => (A,), Field => (C,)), allocator=@rand)
                @test keys(A) == (:in, :out)
                @test !all(A.in .== 0.0)
                @test !all(A.out .== 0.0)
            end;

            @reset_parallel_stencil()
        end;

        @testset "4. double_buffered kwarg for array allocators" begin
            @require !@is_initialized()
            @init_parallel_stencil($package, $FloatDefault, 3, nonconst_metadata=true)
            @require @is_initialized()

            nxyz = (8, 8, 8)

            @testset "@zeros" begin
                result_default = @zeros(nxyz...)
                @test !(result_default isa NamedTuple)
                result = @zeros(nxyz..., double_buffered=true)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out) == nxyz
                @test all(Array(result.in) .== 0.0)
                @test all(Array(result.out) .== 0.0)
            end;

            @testset "@ones" begin
                result_default = @ones(nxyz...)
                @test !(result_default isa NamedTuple)
                result = @ones(nxyz..., double_buffered=true)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out) == nxyz
                @test all(Array(result.in) .== 1.0)
                @test all(Array(result.out) .== 1.0)
                result_f32 = @zeros(nxyz..., eltype=Float32, double_buffered=true)
                @test eltype(result_f32.in) == Float32
                @test eltype(result_f32.out) == Float32
            end;

            @testset "@rand" begin
                result_default = @rand(nxyz...)
                @test !(result_default isa NamedTuple)
                result = @rand(nxyz..., double_buffered=true)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out) == nxyz
            end;

            @testset "@falses" begin
                result_default = @falses(nxyz...)
                @test !(result_default isa NamedTuple)
                result = @falses(nxyz..., double_buffered=true)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out) == nxyz
                @test all(.!Array(result.in))
            end;

            @testset "@trues" begin
                result_default = @trues(nxyz...)
                @test !(result_default isa NamedTuple)
                result = @trues(nxyz..., double_buffered=true)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out) == nxyz
                @test all(Array(result.in))
            end;

            @testset "@fill" begin
                result_default = @fill(3.0, nxyz...)
                @test !(result_default isa NamedTuple)
                result = @fill(3.0, nxyz..., double_buffered=true)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out) == nxyz
                @test all(Array(result.in) .== 3.0)
                @test all(Array(result.out) .== 3.0)
            end;

            @reset_parallel_stencil()
        end;

        @static if $package != $PKG_POLYESTER # TODO: this needs to be removed once Polyester supports padding
        @testset "2. 2B field allocator macros (padding=true)" begin
            @require !@is_initialized()
            @init_parallel_stencil($package, $FloatDefault, 3, padding=true, nonconst_metadata=true)
            @require @is_initialized()

            nxyz = (8, 8, 8)

            @testset "@Field2B" begin
                result = @Field2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test size(result.in) == size(result.out) == nxyz
                @test typeof(result.in) == typeof(@Field(nxyz))
            end;

            @testset "@BVectorField2B" begin
                result = @BVectorField2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test keys(result.in) == (:x, :y, :z)
                @test keys(result.out) == (:x, :y, :z)
                @test size(result.in.x) == size(result.out.x) == size(@BVectorField(nxyz).x)
                @test size(result.in.y) == size(result.out.y) == size(@BVectorField(nxyz).y)
                @test size(result.in.z) == size(result.out.z) == size(@BVectorField(nxyz).z)
            end;

            @testset "@TensorField2B" begin
                result = @TensorField2B(nxyz)
                @test keys(result) == (:in, :out)
                @test result.in !== result.out
                @test keys(result.in) == (:xx, :yy, :zz, :xy, :xz, :yz)
                @test keys(result.out) == (:xx, :yy, :zz, :xy, :xz, :yz)
                @test size(result.in.xx) == size(result.out.xx) == size(@TensorField(nxyz).xx)
            end;

            @reset_parallel_stencil()
        end;
        end

    end;
))

end == nothing || true;

PKG_FOR_XPU_TESTS = (PKG_CUDA in TEST_PACKAGES) ? [PKG_CUDA] : (PKG_AMDGPU in TEST_PACKAGES) ? [PKG_AMDGPU] : (PKG_METAL in TEST_PACKAGES) ? [PKG_METAL] : (PKG_THREADS in TEST_PACKAGES) ? [PKG_THREADS] : []
@static for package in PKG_FOR_XPU_TESTS
    FloatDefault = (package == PKG_METAL) ? Float32 : Float64 # Metal does not support Float64

eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)) - xPU)" begin
        @require !@is_initialized()
        @init_parallel_stencil($package, $FloatDefault, 3, padding=true, nonconst_metadata=true)
        @require @is_initialized()
        using .Data.Fields

        nxyz = (8, 8, 8)

        @testset "2B type aliases" begin
            # All 2B field type aliases are defined in Data.Fields and Data.Fields.Device.
            for T in FIELDTYPES
                @test isdefined(Data.Fields, T)
                @test isdefined(Data.Fields.Device, T)
                @test isdefined(TData.Fields, T)
                @test isdefined(TData.Fields.Device, T)
            end
            # Top-level array type aliases are defined in Data, TData, and their Device submodules.
            @test isdefined(Data, :Array2B)
            @test isdefined(Data, :SubArray2B)
            @test isdefined(Data.Device, :Array2B)
            @test isdefined(Data.Device, :SubArray2B)
            @test isdefined(TData, :Array2B)
            @test isdefined(TData, :SubArray2B)
            @test isdefined(TData.Device, :Array2B)
            @test isdefined(TData.Device, :SubArray2B)
            # NamedTuple covariance: every 2B field allocation must be
            # isa-compatible with its Data.Fields alias (the <:` in the alias
            # definition `const Field2B{N} = NamedTuple{(:in, :out), <:Tuple{...}}`
            # is essential — without it, `Pt isa Field2B` returns `false`).
            @test (@Field2B(nxyz)) isa Data.Fields.Field2B
            @test (@XField2B(nxyz)) isa Data.Fields.XField2B
            @test (@YField2B(nxyz)) isa Data.Fields.YField2B
            @test (@ZField2B(nxyz)) isa Data.Fields.ZField2B
            @test (@BXField2B(nxyz)) isa Data.Fields.BXField2B
            @test (@BYField2B(nxyz)) isa Data.Fields.BYField2B
            @test (@BZField2B(nxyz)) isa Data.Fields.BZField2B
            @test (@XXField2B(nxyz)) isa Data.Fields.XXField2B
            @test (@YYField2B(nxyz)) isa Data.Fields.YYField2B
            @test (@ZZField2B(nxyz)) isa Data.Fields.ZZField2B
            @test (@XYField2B(nxyz)) isa Data.Fields.XYField2B
            @test (@XZField2B(nxyz)) isa Data.Fields.XZField2B
            @test (@YZField2B(nxyz)) isa Data.Fields.YZField2B
            @test (@VectorField2B(nxyz)) isa Data.Fields.VectorField2B
            @test (@BVectorField2B(nxyz)) isa Data.Fields.BVectorField2B
            @test (@TensorField2B(nxyz)) isa Data.Fields.TensorField2B
            # AbstractArray2B covers both Array2B and SubArray2B: in non-padding
            # mode Field=Array so @Field2B is Array2B (AbstractArray2B via
            # Array <: AbstractArray); in padding mode Field=SubArray so
            # @Field2B is SubArray2B (AbstractArray2B via SubArray <: AbstractArray).
            @test (@Field2B(nxyz)) isa Data.AbstractArray2B
        end;
    end;
))

end == nothing || true;
