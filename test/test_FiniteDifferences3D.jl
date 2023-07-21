using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_THREADS
import ParallelStencil: @require
using ParallelStencil.FiniteDifferences3D
TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import ParallelStencil.CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end
@static if PKG_AMDGPU in TEST_PACKAGES
    import ParallelStencil.AMDGPU
    if !AMDGPU.functional() TEST_PACKAGES = filter!(x->x≠PKG_AMDGPU, TEST_PACKAGES) end
end

@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @require !@is_initialized()
        @init_parallel_stencil($package, Float64, 3)
        @require @is_initialized()
        nx, ny, nz = 7, 5, 6
        A       =  @rand(nx  , ny  , nz  );
        Ax      =  @rand(nx+1, ny  , nz  );
        Ay      =  @rand(nx  , ny+1, nz  );
        Az      =  @rand(nx  , ny  , nz+1);
        Axy     =  @rand(nx+1, ny+1, nz  );
        Axz     =  @rand(nx+1, ny  , nz+1);
        Ayz     =  @rand(nx  , ny+1, nz+1);
        Axyz    =  @rand(nx+1, ny+1, nz+1);
        Axyzz   =  @rand(nx+1, ny+1, nz+2);
        Axyyz   =  @rand(nx+1, ny+2, nz+1);
        Axxyz   =  @rand(nx+2, ny+1, nz+1);
        Axx     =  @rand(nx+2, ny  , nz  );
        Ayy     =  @rand(nx  , ny+2, nz  );
        Azz     =  @rand(nx  , ny  , nz+2);
        Axxyy   =  @rand(nx+2, ny+2, nz  );
        Axxzz   =  @rand(nx+2, ny  , nz+2);
        Ayyzz   =  @rand(nx  , ny+2, nz+2);
        Axyyzz  =  @rand(nx+1, ny+2, nz+2);
        Axxyzz  =  @rand(nx+2, ny+1, nz+2);
        Axxyyz  =  @rand(nx+2, ny+2, nz+1);
        Axxyyzz =  @rand(nx+2, ny+2, nz+2);
        R       = @zeros(nx  , ny  , nz  );
        Rxxyyzz = @zeros(nx+2, ny+2, nz+2);
        @testset "1. compute macros" begin
            @testset "differences" begin
                @parallel  d_xa!(R, Ax)      = (@all(R) = @d_xa(Ax); return)
                @parallel  d_ya!(R, Ay)      = (@all(R) = @d_ya(Ay); return)
                @parallel  d_za!(R, Az)      = (@all(R) = @d_za(Az); return)
                @parallel  d_xi!(R, Axyyzz)  = (@all(R) = @d_xi(Axyyzz); return)
                @parallel  d_yi!(R, Axxyzz)  = (@all(R) = @d_yi(Axxyzz); return)
                @parallel  d_zi!(R, Axxyyz)  = (@all(R) = @d_zi(Axxyyz); return)
                @parallel d2_xi!(R, Axxyyzz) = (@all(R) = @d2_xi(Axxyyzz); return)
                @parallel d2_yi!(R, Axxyyzz) = (@all(R) = @d2_yi(Axxyyzz); return)
                @parallel d2_zi!(R, Axxyyzz) = (@all(R) = @d2_zi(Axxyyzz); return)
                R.=0; @parallel  d_xa!(R, Ax);       @test all(Array(R .== Ax[2:end,    :,    :].-Ax[1:end-1,      :,      :]))
                R.=0; @parallel  d_ya!(R, Ay);       @test all(Array(R .== Ay[    :,2:end,    :].-Ay[      :,1:end-1,      :]))
                R.=0; @parallel  d_za!(R, Az);       @test all(Array(R .== Az[    :,    :,2:end].-Az[      :,      :,1:end-1]))
                R.=0; @parallel  d_xi!(R, Axyyzz);   @test all(Array(R .== Axyyzz[2:end  ,2:end-1,2:end-1].-Axyyzz[1:end-1,2:end-1,2:end-1]))
                R.=0; @parallel  d_yi!(R, Axxyzz);   @test all(Array(R .== Axxyzz[2:end-1,2:end  ,2:end-1].-Axxyzz[2:end-1,1:end-1,2:end-1]))
                R.=0; @parallel  d_zi!(R, Axxyyz);   @test all(Array(R .== Axxyyz[2:end-1,2:end-1,2:end  ].-Axxyyz[2:end-1,2:end-1,1:end-1]))
                R.=0; @parallel d2_xi!(R, Axxyyzz);  @test all(Array(R .== (Axxyyzz[3:end,2:end-1,2:end-1].-Axxyyzz[2:end-1,2:end-1,2:end-1]).-(Axxyyzz[2:end-1,2:end-1,2:end-1].-Axxyyzz[1:end-2,2:end-1,2:end-1])))
                R.=0; @parallel d2_yi!(R, Axxyyzz);  @test all(Array(R .== (Axxyyzz[2:end-1,3:end,2:end-1].-Axxyyzz[2:end-1,2:end-1,2:end-1]).-(Axxyyzz[2:end-1,2:end-1,2:end-1].-Axxyyzz[2:end-1,1:end-2,2:end-1])))
                R.=0; @parallel d2_zi!(R, Axxyyzz);  @test all(Array(R .== (Axxyyzz[2:end-1,2:end-1,3:end].-Axxyyzz[2:end-1,2:end-1,2:end-1]).-(Axxyyzz[2:end-1,2:end-1,2:end-1].-Axxyyzz[2:end-1,2:end-1,1:end-2])))
            end;
            @testset "selection" begin
                @parallel all!(R, A)        = (@all(R) = @all(A); return)
                @parallel inn!(R, Axxyyzz)  = (@all(R) = @inn(Axxyyzz); return)
                @parallel inn_x!(R, Axx)    = (@all(R) = @inn_x(Axx); return)
                @parallel inn_y!(R, Ayy)    = (@all(R) = @inn_y(Ayy); return)
                @parallel inn_z!(R, Azz)    = (@all(R) = @inn_z(Azz); return)
                @parallel inn_xy!(R, Axxyy) = (@all(R) = @inn_xy(Axxyy); return)
                @parallel inn_xz!(R, Axxzz) = (@all(R) = @inn_xz(Axxzz); return)
                @parallel inn_yz!(R, Ayyzz) = (@all(R) = @inn_yz(Ayyzz); return)
                R.=0; @parallel all!(R, A);         @test all(Array(R .== A))
                R.=0; @parallel inn!(R, Axxyyzz);   @test all(Array(R .== Axxyyzz[2:end-1,2:end-1,2:end-1]))
                R.=0; @parallel inn_x!(R, Axx);     @test all(Array(R .== Axx[2:end-1,      :,      :]))
                R.=0; @parallel inn_y!(R, Ayy);     @test all(Array(R .== Ayy[      :,2:end-1,      :]))
                R.=0; @parallel inn_z!(R, Azz);     @test all(Array(R .== Azz[      :,      :,2:end-1]))
                R.=0; @parallel inn_xy!(R, Axxyy);  @test all(Array(R .== Axxyy[2:end-1,2:end-1,      :]))
                R.=0; @parallel inn_xz!(R, Axxzz);  @test all(Array(R .== Axxzz[2:end-1,      :,2:end-1]))
                R.=0; @parallel inn_yz!(R, Ayyzz);  @test all(Array(R .== Ayyzz[      :,2:end-1,2:end-1]))
            end;
            @testset "averages" begin
                @parallel av!(R, Axyz)      = (@all(R) = @av(Axyz); return)
                @parallel av_xa!(R, Ax)     = (@all(R) = @av_xa(Ax); return)
                @parallel av_ya!(R, Ay)     = (@all(R) = @av_ya(Ay); return)
                @parallel av_za!(R, Az)     = (@all(R) = @av_za(Az); return)
                @parallel av_xi!(R, Axyyzz) = (@all(R) = @av_xi(Axyyzz); return)
                @parallel av_yi!(R, Axxyzz) = (@all(R) = @av_yi(Axxyzz); return)
                @parallel av_zi!(R, Axxyyz) = (@all(R) = @av_zi(Axxyyz); return)
                @parallel av_xya!(R, Axy)   = (@all(R) = @av_xya(Axy); return)
                @parallel av_xza!(R, Axz)   = (@all(R) = @av_xza(Axz); return)
                @parallel av_yza!(R, Ayz)   = (@all(R) = @av_yza(Ayz); return)
                @parallel av_xyi!(R, Axyzz) = (@all(R) = @av_xyi(Axyzz); return)
                @parallel av_xzi!(R, Axyyz) = (@all(R) = @av_xzi(Axyyz); return)
                @parallel av_yzi!(R, Axxyz) = (@all(R) = @av_yzi(Axxyz); return)
                R.=0; @parallel av!(R, Axyz);       @test all(Array(R .== (Axyz[1:end-1,1:end-1,1:end-1].+Axyz[2:end,1:end-1,1:end-1].+Axyz[2:end,2:end,1:end-1].+Axyz[2:end,2:end,2:end].+Axyz[1:end-1,2:end,2:end].+Axyz[1:end-1,1:end-1,2:end].+Axyz[2:end,1:end-1,2:end].+Axyz[1:end-1,2:end,1:end-1])*0.125))
                R.=0; @parallel av_xa!(R, Ax);      @test all(Array(R .== (Ax[2:end,    :,    :].+Ax[1:end-1,      :,    :]).*0.5))
                R.=0; @parallel av_ya!(R, Ay);      @test all(Array(R .== (Ay[    :,2:end,    :].+Ay[      :,1:end-1,    :]).*0.5))
                R.=0; @parallel av_za!(R, Az);      @test all(Array(R .== (Az[    :,    :,2:end].+Az[      :,    :,1:end-1]).*0.5))
                R.=0; @parallel av_xi!(R, Axyyzz);  @test all(Array(R .== (Axyyzz[2:end  ,2:end-1,2:end-1].+Axyyzz[1:end-1,2:end-1,2:end-1]).*0.5))
                R.=0; @parallel av_yi!(R, Axxyzz);  @test all(Array(R .== (Axxyzz[2:end-1,2:end  ,2:end-1].+Axxyzz[2:end-1,1:end-1,2:end-1]).*0.5))
                R.=0; @parallel av_zi!(R, Axxyyz);  @test all(Array(R .== (Axxyyz[2:end-1,2:end-1,2:end  ].+Axxyyz[2:end-1,2:end-1,1:end-1]).*0.5))
                R.=0; @parallel av_xya!(R, Axy);    @test all(Array(R .== (Axy[1:end-1,1:end-1,:].+Axy[2:end,1:end-1,:].+Axy[1:end-1,2:end,:].+Axy[2:end,2:end,:])*0.25))
                R.=0; @parallel av_xza!(R, Axz);    @test all(Array(R .== (Axz[1:end-1,:,1:end-1].+Axz[2:end,:,1:end-1].+Axz[1:end-1,:,2:end].+Axz[2:end,:,2:end])*0.25))
                R.=0; @parallel av_yza!(R, Ayz);    @test all(Array(R .== (Ayz[:,1:end-1,1:end-1].+Ayz[:,2:end,1:end-1].+Ayz[:,1:end-1,2:end].+Ayz[:,2:end,2:end])*0.25))
                R.=0; @parallel av_xyi!(R, Axyzz);  @test all(Array(R .== (Axyzz[1:end-1,1:end-1,2:end-1].+Axyzz[2:end,1:end-1,2:end-1].+Axyzz[1:end-1,2:end,2:end-1].+Axyzz[2:end,2:end,2:end-1])*0.25))
                R.=0; @parallel av_xzi!(R, Axyyz);  @test all(Array(R .== (Axyyz[1:end-1,2:end-1,1:end-1].+Axyyz[2:end,2:end-1,1:end-1].+Axyyz[1:end-1,2:end-1,2:end].+Axyyz[2:end,2:end-1,2:end])*0.25))
                R.=0; @parallel av_yzi!(R, Axxyz);  @test all(Array(R .== (Axxyz[2:end-1,1:end-1,1:end-1].+Axxyz[2:end-1,2:end,1:end-1].+Axxyz[2:end-1,1:end-1,2:end].+Axxyz[2:end-1,2:end,2:end])*0.25))
            end;
            @testset "harmonic averages" begin
                @parallel harm!(R, Axyz)      = (@all(R) = @harm(Axyz); return)
                @parallel harm_xa!(R, Ax)     = (@all(R) = @harm_xa(Ax); return)
                @parallel harm_ya!(R, Ay)     = (@all(R) = @harm_ya(Ay); return)
                @parallel harm_za!(R, Az)     = (@all(R) = @harm_za(Az); return)
                @parallel harm_xi!(R, Axyyzz) = (@all(R) = @harm_xi(Axyyzz); return)
                @parallel harm_yi!(R, Axxyzz) = (@all(R) = @harm_yi(Axxyzz); return)
                @parallel harm_zi!(R, Axxyyz) = (@all(R) = @harm_zi(Axxyyz); return)
                @parallel harm_xya!(R, Axy)   = (@all(R) = @harm_xya(Axy); return)
                @parallel harm_xza!(R, Axz)   = (@all(R) = @harm_xza(Axz); return)
                @parallel harm_yza!(R, Ayz)   = (@all(R) = @harm_yza(Ayz); return)
                @parallel harm_xyi!(R, Axyzz) = (@all(R) = @harm_xyi(Axyzz); return)
                @parallel harm_xzi!(R, Axyyz) = (@all(R) = @harm_xzi(Axyyz); return)
                @parallel harm_yzi!(R, Axxyz) = (@all(R) = @harm_yzi(Axxyz); return)
                R.=0; @parallel harm!(R, Axyz);       @test all(Array(R .== 8 ./(1 ./Axyz[1:end-1,1:end-1,1:end-1].+1 ./Axyz[2:end,1:end-1,1:end-1].+1 ./Axyz[2:end,2:end,1:end-1].+1 ./Axyz[2:end,2:end,2:end].+1 ./Axyz[1:end-1,2:end,2:end].+1 ./Axyz[1:end-1,1:end-1,2:end].+1 ./Axyz[2:end,1:end-1,2:end].+1 ./Axyz[1:end-1,2:end,1:end-1]) ))
                R.=0; @parallel harm_xa!(R, Ax);      @test all(Array(R .== 2 ./(1 ./Ax[2:end,    :,    :].+1 ./Ax[1:end-1,      :,    :])))
                R.=0; @parallel harm_ya!(R, Ay);      @test all(Array(R .== 2 ./(1 ./Ay[    :,2:end,    :].+1 ./Ay[      :,1:end-1,    :])))
                R.=0; @parallel harm_za!(R, Az);      @test all(Array(R .== 2 ./(1 ./Az[    :,    :,2:end].+1 ./Az[      :,    :,1:end-1])))
                R.=0; @parallel harm_xi!(R, Axyyzz);  @test all(Array(R .== 2 ./(1 ./Axyyzz[2:end  ,2:end-1,2:end-1].+1 ./Axyyzz[1:end-1,2:end-1,2:end-1])))
                R.=0; @parallel harm_yi!(R, Axxyzz);  @test all(Array(R .== 2 ./(1 ./Axxyzz[2:end-1,2:end  ,2:end-1].+1 ./Axxyzz[2:end-1,1:end-1,2:end-1])))
                R.=0; @parallel harm_zi!(R, Axxyyz);  @test all(Array(R .== 2 ./(1 ./Axxyyz[2:end-1,2:end-1,2:end  ].+1 ./Axxyyz[2:end-1,2:end-1,1:end-1])))
                R.=0; @parallel harm_xya!(R, Axy);    @test all(Array(R .== 4 ./(1 ./Axy[1:end-1,1:end-1,:].+1 ./Axy[2:end,1:end-1,:].+1 ./Axy[1:end-1,2:end,:].+1 ./Axy[2:end,2:end,:])))
                R.=0; @parallel harm_xza!(R, Axz);    @test all(Array(R .== 4 ./(1 ./Axz[1:end-1,:,1:end-1].+1 ./Axz[2:end,:,1:end-1].+1 ./Axz[1:end-1,:,2:end].+1 ./Axz[2:end,:,2:end])))
                R.=0; @parallel harm_yza!(R, Ayz);    @test all(Array(R .== 4 ./(1 ./Ayz[:,1:end-1,1:end-1].+1 ./Ayz[:,2:end,1:end-1].+1 ./Ayz[:,1:end-1,2:end].+1 ./Ayz[:,2:end,2:end])))
                R.=0; @parallel harm_xyi!(R, Axyzz);  @test all(Array(R .== 4 ./(1 ./Axyzz[1:end-1,1:end-1,2:end-1].+1 ./Axyzz[2:end,1:end-1,2:end-1].+1 ./Axyzz[1:end-1,2:end,2:end-1].+1 ./Axyzz[2:end,2:end,2:end-1])))
                R.=0; @parallel harm_xzi!(R, Axyyz);  @test all(Array(R .== 4 ./(1 ./Axyyz[1:end-1,2:end-1,1:end-1].+1 ./Axyyz[2:end,2:end-1,1:end-1].+1 ./Axyyz[1:end-1,2:end-1,2:end].+1 ./Axyyz[2:end,2:end-1,2:end])))
                R.=0; @parallel harm_yzi!(R, Axxyz);  @test all(Array(R .== 4 ./(1 ./Axxyz[2:end-1,1:end-1,1:end-1].+1 ./Axxyz[2:end-1,2:end,1:end-1].+1 ./Axxyz[2:end-1,1:end-1,2:end].+1 ./Axxyz[2:end-1,2:end,2:end])))
            end;
            @testset "others" begin
                @parallel maxloc!(R, Axxyyzz) = (@all(R) = @maxloc(Axxyyzz); return)
                R.=0; @parallel maxloc!(R, Axxyyzz); @test all(Array(R .== max.(max.(max.(max.(max.(max.(Axxyyzz[1:end-2,2:end-1,2:end-1],Axxyyzz[3:end,2:end-1,2:end-1]),Axxyyzz[2:end-1,2:end-1,2:end-1]),Axxyyzz[2:end-1,1:end-2,2:end-1]),Axxyyzz[2:end-1,3:end,2:end-1]),Axxyyzz[2:end-1,2:end-1,1:end-2]),Axxyyzz[2:end-1,2:end-1,3:end])))
            end;
        end;
        @testset "2. apply masks" begin
            @testset "selection" begin
                @parallel inn_all!(Rxxyyzz, A)       = (@inn(Rxxyyzz) = @all(A); return)
                @parallel inn_inn!(Rxxyyzz, Axxyyzz) = (@inn(Rxxyyzz) = @inn(Axxyyzz); return)
                Rxxyyzz.=0; @parallel inn_all!(Rxxyyzz, A);        @test all(Array(Rxxyyzz[2:end-1,2:end-1,2:end-1] .== A))
                Rxxyyzz[2:end-1,2:end-1,2:end-1].=0;               @test all(Array(Rxxyyzz .== 0))  # Test that boundary values remained zero.
                Rxxyyzz.=0; @parallel inn_inn!(Rxxyyzz, Axxyyzz);  @test all(Array(Rxxyyzz[2:end-1,2:end-1,2:end-1] .== Axxyyzz[2:end-1,2:end-1,2:end-1]))
                Rxxyyzz[2:end-1,2:end-1,2:end-1].=0;               @test all(Array(Rxxyyzz .== 0))  # Test that boundary values remained zero.
            end;
            @testset "differences" begin
                @parallel  inn_d_xa!(Rxxyyzz, Ax)       = (@inn(Rxxyyzz) = @d_xa(Ax); return)
                @parallel  inn_d_yi!(Rxxyyzz, Axxyzz)   = (@inn(Rxxyyzz) = @d_yi(Axxyzz); return)
                @parallel  inn_d2_yi!(Rxxyyzz, Axxyyzz) = (@inn(Rxxyyzz) = @d2_yi(Axxyyzz); return)
                Rxxyyzz.=0; @parallel  inn_d_xa!(Rxxyyzz, Ax);       @test all(Array(Rxxyyzz[2:end-1,2:end-1,2:end-1] .== Ax[2:end,    :,    :].-Ax[1:end-1,      :,    :]))
                Rxxyyzz[2:end-1,2:end-1,2:end-1].=0;                 @test all(Array(Rxxyyzz .== 0))  # Test that boundary values remained zero.
                Rxxyyzz.=0; @parallel  inn_d_yi!(Rxxyyzz, Axxyzz);   @test all(Array(Rxxyyzz[2:end-1,2:end-1,2:end-1] .== Axxyzz[2:end-1,2:end  ,2:end-1].-Axxyzz[2:end-1,1:end-1,2:end-1]))
                Rxxyyzz[2:end-1,2:end-1,2:end-1].=0;                 @test all(Array(Rxxyyzz .== 0))  # Test that boundary values remained zero.
                Rxxyyzz.=0; @parallel inn_d2_yi!(Rxxyyzz, Axxyyzz);  @test all(Array(Rxxyyzz[2:end-1,2:end-1,2:end-1] .== (Axxyyzz[2:end-1,3:end,2:end-1].-Axxyyzz[2:end-1,2:end-1,2:end-1]).-(Axxyyzz[2:end-1,2:end-1,2:end-1].-Axxyyzz[2:end-1,1:end-2,2:end-1])))
                Rxxyyzz[2:end-1,2:end-1,2:end-1].=0;                 @test all(Array(Rxxyyzz .== 0))  # Test that boundary values remained zero.
            end;
        end;
        @reset_parallel_stencil()
    end;
)) end == nothing || true;
