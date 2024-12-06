using Pkg, Test
using Revise

package_root  = joinpath(@__DIR__, "..")
test_project  = joinpath(@__DIR__, "test_projects", "Diffusion3D_Revise")
test_file     = joinpath(@__DIR__, "test_projects", "shared", "diffusion3D.jl")
test_file_tmp = joinpath(@__DIR__, "test_projects", "Diffusion3D_Revise", "src", "diffusion3D_tmp.jl")

# Copy the missing test file to the test project, activate, use and test it
cp(test_file, test_file_tmp; force=true)
Pkg.activate(test_project)
# Pkg.instantiate()
Pkg.develop(path=package_root)
using Diffusion3D_Revise
@test Diffusion3D_Revise.diffusion3D()

# Revise the test file in the test project and test it
test_code = read(test_file, String)
write(test_file_tmp, replace(test_code, "return true"=>"return false"))
@test !(Diffusion3D_Revise.diffusion3D())
