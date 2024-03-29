# https://github.com/fastai/fastai1/blob/master/fastai/utils/collect_env.py

import torch
import fastai
import fastcore
import os
import sys

def try_import(module):
    "Try to import `module`. Returns module's object on success, None on failure"
    try: return importlib.import_module(module)
    except: return None

def get_env(name):
    "Return env var value if it's defined and not an empty string, or return Unknown"
    res = os.environ.get(name,'')
    return res if len(res) else "Unknown"


def show_install(show_nvidia_smi:bool=False):
    "Print user's setup information"

    import platform

    rep = []
    opt_mods = []

    rep.append(["=== Software ===", None])
    rep.append(["python", platform.python_version()])
    rep.append(["fastai", fastai.__version__])
    # rep.append(["fastprogress", fastprogress.__version__])
    rep.append(["fastcore", fastcore.__version__])
    rep.append(["torch",  torch.__version__])

    # nvidia-smi
    cmd = "nvidia-smi"
    have_nvidia_smi = False
    try: result = subprocess.run(cmd.split(), shell=False, check=False, stdout=subprocess.PIPE)
    except: pass
    else:
        if result.returncode == 0 and result.stdout: have_nvidia_smi = True

    # XXX: if nvidia-smi is not available, another check could be:
    # /proc/driver/nvidia/version on most systems, since it's the
    # currently active version

    if have_nvidia_smi:
        smi = result.stdout.decode('utf-8')
        # matching: "Driver Version: 396.44"
        match = re.findall(r'Driver Version: +(\d+\.\d+)', smi)
        if match: rep.append(["nvidia driver", match[0]])

    available = "available" if torch.cuda.is_available() else "**Not available** "
    # try:
    rep.append(["torch cuda", f"{torch.version.cuda} / is {available}"])
    # except:
        # rep.append(["torch cuda", f"{torch.__version__} / is {available}"])

    # no point reporting on cudnn if cuda is not available, as it
    # seems to be enabled at times even on cpu-only setups
    if torch.cuda.is_available():
        enabled = "enabled" if torch.backends.cudnn.enabled else "**Not enabled** "
        rep.append(["torch cudnn", f"{torch.backends.cudnn.version()} / is {enabled}"])

    rep.append(["\n=== Hardware ===", None])

    # it's possible that torch might not see what nvidia-smi sees?
    gpu_total_mem = []
    nvidia_gpu_cnt = 0
    if have_nvidia_smi:
        try:
            cmd = "nvidia-smi --query-gpu=memory.total --format=csv,nounits,noheader"
            result = subprocess.run(cmd.split(), shell=False, check=False, stdout=subprocess.PIPE)
        except:
            print("have nvidia-smi, but failed to query it")
        else:
            if result.returncode == 0 and result.stdout:
                output = result.stdout.decode('utf-8')
                gpu_total_mem = [int(x) for x in output.strip().split('\n')]
                nvidia_gpu_cnt = len(gpu_total_mem)


    if nvidia_gpu_cnt: rep.append(["nvidia gpus", nvidia_gpu_cnt])

    torch_gpu_cnt = torch.cuda.device_count()
    if torch_gpu_cnt:
        rep.append(["torch devices", torch_gpu_cnt])
        # information for each gpu
        for i in range(torch_gpu_cnt):
            rep.append([f"  - gpu{i}", (f"{gpu_total_mem[i]}MB | " if gpu_total_mem else "") + torch.cuda.get_device_name(i)])
    else:
        if nvidia_gpu_cnt:
            rep.append([f"Have {nvidia_gpu_cnt} GPU(s), but torch can't use them (check nvidia driver)", None])
        else:
            rep.append([f"No GPUs available", None])


    rep.append(["\n=== Environment ===", None])

    rep.append(["platform", platform.platform()])

    if platform.system() == 'Linux':
        distro = try_import('distro')
        if distro:
            # full distro info
            rep.append(["distro", ' '.join(distro.linux_distribution())])
        else:
            opt_mods.append('distro');
            # partial distro info
            rep.append(["distro", platform.uname().version])

    rep.append(["conda env", get_env('CONDA_DEFAULT_ENV')])
    rep.append(["python", sys.executable])
    rep.append(["sys.path", "\n".join(sys.path)])

    print("\n\n```text")

    keylen = max([len(e[0]) for e in rep if e[1] is not None])
    for e in rep:
        print(f"{e[0]:{keylen}}", (f": {e[1]}" if e[1] is not None else ""))

    if have_nvidia_smi:
        if show_nvidia_smi: print(f"\n{smi}")
    else:
        if torch_gpu_cnt: print("no nvidia-smi is found")
        else: print("no supported gpus found on this system")

    print("```\n")

    print("Please make sure to include opening/closing ``` when you paste into forums/github to make the reports appear formatted as code sections.\n")

    if opt_mods:
        print("Optional package(s) to enhance the diagnostics can be installed with:")
        print(f"pip install {' '.join(opt_mods)}")
        print("Once installed, re-run this utility to get the additional information")
