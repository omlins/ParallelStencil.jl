# Selection of the packages (backends) to test, shared by all test files: `TEST_PACKAGES` contains the supported packages that are
# installed in the test environment and functional; they are imported into Main for the test files.
# NOTE: the GPU packages are not dependencies of test/Project.toml, as resolving them all together holds each of them back to an old
# version (or fails altogether); the test runner (runtests.jl) installs the one given by PS_TEST_BACKEND into the test environment (see
# the Testing page of the documentation). The test files of the test projects (test/test_projects) are only run for the packages in
# `TEST_PACKAGES`.
import ParallelStencil
import ParallelStencil: SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_KERNELABSTRACTIONS, PKG_POLYESTER

is_installed(package::Symbol) = !isnothing(Base.find_package(string(package)))

# Import the GPU packages that are installed; the others are replaced by empty modules of the same name, as the test files refer to the
# packages by their names (e.g. `$package == $PKG_CUDA` compares the modules `Threads` and `CUDA`).
for package in (PKG_CUDA, PKG_AMDGPU, PKG_METAL)
    if is_installed(package) Core.eval(@__MODULE__, :(import $package))
    else                     Core.eval(@__MODULE__, :(module $package end))
    end
end
import KernelAbstractions, Polyester
const TEST_AD = is_installed(:Enzyme) # Whether the tests of automatic differentiation are run (Enzyme is installed by the test runner unless PS_TEST_AD=0).
@static if TEST_AD import Enzyme end

TEST_PACKAGES = copy(SUPPORTED_PACKAGES)
if (PKG_CUDA               in TEST_PACKAGES && !(is_installed(PKG_CUDA)   && CUDA.functional()))        filter!(x->x≠PKG_CUDA,               TEST_PACKAGES) end
if (PKG_AMDGPU             in TEST_PACKAGES && !(is_installed(PKG_AMDGPU) && AMDGPU.functional()))      filter!(x->x≠PKG_AMDGPU,             TEST_PACKAGES) end
if (PKG_METAL              in TEST_PACKAGES && !(is_installed(PKG_METAL)  && Metal.functional()))       filter!(x->x≠PKG_METAL,              TEST_PACKAGES) end
if (PKG_KERNELABSTRACTIONS in TEST_PACKAGES && !KernelAbstractions.functional(KernelAbstractions.CPU())) filter!(x->x≠PKG_KERNELABSTRACTIONS, TEST_PACKAGES) end
Base.retry_load_extensions() # Potentially needed to load the extensions after the packages have been filtered.

# Hardware symbols of the GPU packages to test with the KernelAbstractions package.
const KA_GPU_SYMBOLS = filter(!isnothing, (
    PKG_CUDA   in TEST_PACKAGES ? :gpu_cuda  : nothing,
    PKG_AMDGPU in TEST_PACKAGES ? :gpu_amd   : nothing,
    PKG_METAL  in TEST_PACKAGES ? :gpu_metal : nothing,
    (isdefined(ParallelStencil.ParallelKernel, :PKG_ONEAPI) && ParallelStencil.ParallelKernel.PKG_ONEAPI in TEST_PACKAGES) ? :gpu_oneapi : nothing
))
