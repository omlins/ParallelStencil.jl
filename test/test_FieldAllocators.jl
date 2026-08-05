using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_THREADS, PKG_POLYESTER, PKG_KERNELABSTRACTIONS
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

            @testset "2B type aliases" begin
                # NOTE: `Data` is created by @init_parallel_stencil via @eval(caller, ...baremodule Data ...),
                # but the `Data` symbol binding inside this eval(:( @testset ...)) was captured BEFORE
                # @init_parallel_stencil ran, so it refers to the stale empty module. We use bare
                # eval(:(expr)) to resolve `Data` at runtime in Main (where @init_parallel_stencil
                # created it), rather than the captured binding.
                @test eval(:(isdefined(Main, :Data)))
                @test eval(:(isdefined(Data, :Array2B)))
                @test eval(:(isdefined(Data, :SubArray2B)))
                @static if $package != $PKG_KERNELABSTRACTIONS
                    @test eval(:(isdefined(Data.Device, :Array2B)))
                    @test eval(:(isdefined(Data.Device, :SubArray2B)))
                end
                @test eval(:(isdefined(Main, :TData)))
                @test eval(:(isdefined(TData, :Array2B)))
                @test eval(:(isdefined(TData, :SubArray2B)))
                @static if $package != $PKG_KERNELABSTRACTIONS
                    @test eval(:(isdefined(TData.Device, :Array2B)))
                    @test eval(:(isdefined(TData.Device, :SubArray2B)))
                end
                @test eval(:(isdefined(Data.Fields, :Field2B)))
                @test eval(:(isdefined(Data.Fields, :XField2B)))
                @test eval(:(isdefined(Data.Fields, :YField2B)))
                @test eval(:(isdefined(Data.Fields, :ZField2B)))
                @test eval(:(isdefined(Data.Fields, :BXField2B)))
                @test eval(:(isdefined(Data.Fields, :BYField2B)))
                @test eval(:(isdefined(Data.Fields, :BZField2B)))
                @test eval(:(isdefined(Data.Fields, :XXField2B)))
                @test eval(:(isdefined(Data.Fields, :YYField2B)))
                @test eval(:(isdefined(Data.Fields, :ZZField2B)))
                @test eval(:(isdefined(Data.Fields, :XYField2B)))
                @test eval(:(isdefined(Data.Fields, :XZField2B)))
                @test eval(:(isdefined(Data.Fields, :YZField2B)))
                @test eval(:(isdefined(Data.Fields, :VectorField2B)))
                @test eval(:(isdefined(Data.Fields, :BVectorField2B)))
                @test eval(:(isdefined(Data.Fields, :TensorField2B)))
                @static if $package != $PKG_KERNELABSTRACTIONS
                    @test eval(:(isdefined(Data.Fields.Device, :Field2B)))
                    @test eval(:(isdefined(Data.Fields.Device, :XField2B)))
                    @test eval(:(isdefined(Data.Fields.Device, :BVectorField2B)))
                    @test eval(:(isdefined(Data.Fields.Device, :TensorField2B)))
                end
            end;

            @testset "2B field isa checks (NamedTuple covariance fix)" begin
                result = @Field2B(nxyz)
                @test eval(:(result isa Data.Fields.Field2B))
                @static if $package != $PKG_KERNELABSTRACTIONS
                    @test eval(:(result isa Data.Fields.Device.Field2B))
                end
                result = @BVectorField2B(nxyz)
                @test eval(:(result isa Data.Fields.BVectorField2B))
                @static if $package != $PKG_KERNELABSTRACTIONS
                    @test eval(:(result isa Data.Fields.Device.BVectorField2B))
                end
                result = @TensorField2B(nxyz)
                @test eval(:(result isa Data.Fields.TensorField2B))
                @static if $package != $PKG_KERNELABSTRACTIONS
                    @test eval(:(result isa Data.Fields.Device.TensorField2B))
                end
                result = @XField2B(nxyz)
                @test eval(:(result isa Data.Fields.XField2B))
                result = @BZField2B(nxyz)
                @test eval(:(result isa Data.Fields.BZField2B))
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
