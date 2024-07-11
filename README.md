# CUDA Lab

## Vultr GPU Enabled VM Setup

1. `apt update`
1. `apt upgrade`
1. `git clone https://github.com/chenbobby/cuda_lab`
1. Open `cuda_lab` repository in VS Code.
1. Install recommended extensions.
    1. The `clangd` VS Code extension should prompt you to install `clangd`.
1. `apt install nvidia-cuda-toolkit`

## Performance Profiling Setup
1. Ensure that the `perf_event_paranoid` level is <= 2.
    1. Check via `cat /proc/sys/kernel/perf_event_paranoid`
    1. Update via `echo 2 > /proc/sys/kernel/perf_event_paranoid`
    1. Configure reboot via `echo kernel.perf_event_paranoid=2 > /etc/sysctl.d/local.conf`
1. Ensure that Ubuntu Linux distro is >= 4.3
    1. Check via `uname -a`
1. Ensure that `glibc` version is >= 2.17
    1. Check via `ldd --version`
1. Ensure that CUDA version and Driver version are compatible, according to [NVIDIA docs](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html#cuda-version).
    1. Check via `nvidia-smi`
1. Add GPG key for APT repository.
    1. `apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub`
1. Add APT repository.
    1. `add-apt-repository "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/ /"`
1. Install `nsight-systems-cli`
    1. Install via `apt install nsight-systems-cli`
    1. You may need to fix broken installs via `apt --fix-broken install`
1. Check your installation.
    1. Check via `nsys status --environment`