"""
    @reset_parallel_kernel()

Reset the ParallelKernel module.

See also: [`init_parallel_kernel`](@ref)
"""
macro reset_parallel_kernel() esc(reset_parallel_kernel(__module__)) end

function reset_parallel_kernel(caller::Module)
    if isdefined(caller, :Data) && isdefined(caller.Data, :Device) # "Clear" the Data module if it has been created by ParallelKernel (i.e. contains Data.Device).
        data_module = Data_none()
        @eval(caller, $data_module)
    end
    set_initialized(caller, false)
    set_package(caller, PKG_NONE)
    set_numbertype(caller, NUMBERTYPE_NONE)
    return nothing
end
