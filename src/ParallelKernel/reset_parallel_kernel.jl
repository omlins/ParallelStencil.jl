"""
    @reset_parallel_kernel()

Reset the ParallelKernel module.

See also: [`init_parallel_kernel`](@ref)
"""
macro reset_parallel_kernel() esc(reset_parallel_kernel(__module__)) end

function reset_parallel_kernel(caller::Module)
    if isdefined(caller, :Data) && isdefined(caller.Data, :DeviceArray) # "Clear" the Data module if it has been created by ParallelKernel (i.e. contains Data.DeviceArray).
        data_module = Data_none()
        @eval(caller, $data_module)
    end
    set_initialized(false)
    set_package(PKG_NONE)
    set_numbertype(NUMBERTYPE_NONE)
    return nothing
end
