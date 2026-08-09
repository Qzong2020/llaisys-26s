-- xmake/nvidia.lua
--
-- CUDA-enabled build flow for LLAISYS (enabled with `--nv-gpu=y`).
--
-- Platform A (NVIDIA): Xmake's built-in CUDA machinery drives `nvcc` against a real CUDA
-- SDK that Xmake auto-detects (`/usr/local/cuda*` or an explicit `--cuda=...`).
--
-- Platform B (MetaX MACA, cu-bridge): there is no `nvcc`. We synthesize an idempotent,
-- minimal "CUDA SDK" shim under `$HOME/.cache/llaisys/maca-cuda/` whose `bin/nvcc` is the
-- MACA `cucc` wrapper (it translates nvcc-style flags into `mxcc -x maca`). The same
-- built-in CUDA machinery then drives `mxcc`, and we additionally link the MACA runtime
-- (`libsymbol_cu` / `libruntime_cu`). This keeps one logical `LLAISYS_DEVICE_NVIDIA`
-- backend for both platforms with no new device type or vendor branch.

option("maca-cuda")
    set_showmenu(false)
    on_check(function ()
        import("core.project.config")

        -- respect an explicitly configured CUDA SDK (--cuda=...)
        local configured = config.get("cuda")
        if configured and os.isexec(path.join(path.join(configured, "bin"), "nvcc")) then
            return
        end

        -- otherwise, if the MetaX MACA cu-bridge is present, build the shim CUDA SDK
        local maca_path = os.getenv("MACA_PATH") or "/opt/maca"
        local maca_llvm = os.getenv("MACA_CLANG_PATH") or path.join(maca_path, "mxgpu_llvm", "bin")
        local cu_bridge = path.join(maca_path, "tools", "cu-bridge")
        local cucc = path.join(cu_bridge, "bin", "cucc")
        if not (os.isexec(path.join(maca_llvm, "mxcc"))
            and os.isdir(path.join(cu_bridge, "include"))
            and os.isfile(cucc)) then
            return
        end

        local shim = path.join(os.getenv("HOME") or os.tmpdir(), ".cache", "llaisys", "maca-cuda")
        os.mkdir(path.join(shim, "bin"))
        os.mkdir(path.join(shim, "lib64"))

        local nvcc_shim = path.join(shim, "bin", "nvcc")
        if not os.isfile(nvcc_shim) then
            local f = io.open(nvcc_shim, "w")
            f:write(string.format([[#!/bin/bash
export MACA_PATH="%s"
export MACA_CLANG_PATH="%s"
exec "%s" "$@"
]], maca_path, maca_llvm, cucc))
            f:close()
            os.exec("chmod +x " .. nvcc_shim)
        end
        if not os.exists(path.join(shim, "include")) then
            os.exec("ln -sfn " .. path.join(cu_bridge, "include") .. " " .. path.join(shim, "include"))
        end
        if not os.isfile(path.join(shim, "lib64", "libcudart_static.a")) then
            -- Xmake's built-in CUDA rule adds -lcudart_static/-lcudadevrt; empty archives
            -- satisfy the link line, the real runtime symbols come from libsymbol_cu etc.
            os.exec("ar rcs " .. path.join(shim, "lib64", "libcudart_static.a"))
            os.exec("ar rcs " .. path.join(shim, "lib64", "libcudadevrt.a"))
        end

        config.set("cuda", shim, {force = true, readonly = true})
        config.set("maca_cu_bridge", true, {force = true})
    end)
option_end()

-- Link the active platform's CUDA runtime/BLAS dependencies through both static targets.
local function add_cuda_runtime_links(target)
    import("core.project.config")
    if config.get("maca_cu_bridge") then
        local libdir = path.join(os.getenv("MACA_PATH") or "/opt/maca", "lib")
        target:add("linkdirs", libdir, {public = true})
        target:add("links", "symbol_cu", "runtime_cu", {public = true})
        target:add("rpathdirs", libdir, {public = true})
    else
        target:add("links", "cublas", {public = true})
    end
end

target("llaisys-device-nvidia")
    set_kind("static")
    set_values("cuda.rdc", false)
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/nvidia/*.cu")
    add_cuflags("-Xcompiler -fPIC", {force = true})

    on_config(add_cuda_runtime_links)
    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    set_values("cuda.rdc", false)
    add_cugencodes("sm_80")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/ops/*/nvidia/*.cu")
    add_cuflags("-Xcompiler -fPIC", {force = true})

    on_config(add_cuda_runtime_links)
    on_install(function (target) end)
target_end()
