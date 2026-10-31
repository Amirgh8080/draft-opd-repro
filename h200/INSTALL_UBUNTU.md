# Fresh Ubuntu on an H200 machine: install everything

This guide starts right after a clean Ubuntu install on a machine with an H200,
and ends with the machine ready for the Draft-OPD runs. That means the NVIDIA
driver, the disk, the CUDA toolkit, Python and every package, all installed and
checked. After that, [`GUIDE_LINUX.md`](GUIDE_LINUX.md) takes over from its
Step 5.

Work through it top to bottom. Each part ends with a check; don't continue
until it passes. All commands run on the H200 machine, in a terminal or over
SSH.

| Part | What it installs | Reboot |
|---|---|---|
| 1 | OS updates and base tools | yes |
| 2 | Identify your H200 system type | |
| 3 | NVIDIA driver (+ Fabric Manager on HGX systems only) | yes |
| 4 | Protections for long runs: driver pinning, no sleep | |
| 5 | Data disk for the workspace | |
| 6 | Everything the project needs: CUDA toolkit, Python 3.12, packages, flash-attn | |
| 7 | Final checklist | |

**Supported:** Ubuntu **24.04 LTS** (recommended) or 22.04 LTS, x86_64,
Desktop or Server.

```bash
lsb_release -ds; uname -m          # e.g. "Ubuntu 24.04.x LTS" and "x86_64"
```

---

## Part 0. Firmware settings (only if Part 3's check fails)

Most H200 servers ship with these already set. Come back here only if
`nvidia-smi` reports *No devices were found* after Part 3:

- **Above 4G Decoding: Enabled.** Data center GPUs map large memory regions
  and need this.
- **Re-Size BAR Support: Enabled**, if your firmware offers it.
- **Secure Boot** can stay on. Part 3 uses Canonical-signed driver modules.

The H200 has **no display output**. Connect the monitor to the motherboard's
own video port, or use the server's remote console (BMC/IPMI).

---

## Part 1. Update the system and install base tools

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y openssh-server build-essential git curl wget tmux htop \
  pciutils ubuntu-drivers-common
sudo reboot
```

`openssh-server` lets you log in remotely, which is useful for weeks-long
training. After the reboot, find the machine's address with `ip -br addr` and
connect from another computer with `ssh <user>@<address>`. From Windows, see
[`GUIDE_WINDOWS.md`](GUIDE_WINDOWS.md) Route 1.

**Check:** `uname -r` prints the kernel you just rebooted into.

---

## Part 2. Identify your H200 system type

```bash
lspci -nn | grep -i nvidia
```

- **H200 NVL / PCIe card:** you see one line per GPU, for example
  `3D controller: NVIDIA Corporation GH100 [H200 NVL]`, and **no**
  `Bridge` lines. Fabric Manager is **not needed**.
- **HGX H200 (SXM baseboard, usually 4 or 8 GPUs):** you also see several
  `Bridge: NVIDIA Corporation ...` lines. Those are the NVSwitches, and
  Fabric Manager is **required**. Without it, CUDA fails with *error 802:
  system not yet initialized*.

If no NVIDIA line appears at all, the card is not detected by the hardware.
Check that it is seated and powered, and check the Part 0 firmware settings.

Write down which one you have. Part 3 asks.

---

## Part 3. Install the NVIDIA driver

Use Ubuntu's own packaged data center driver (the **`-server`** branch)
through `ubuntu-drivers`. It installs prebuilt kernel modules signed by
Canonical, which work with Secure Boot and survive kernel updates. Do **not**
use NVIDIA's `.run` installer, and don't mix in driver packages from other
sources.

### 3.1 See which drivers are available

```bash
sudo ubuntu-drivers list --gpgpu
```

It prints names like `nvidia-driver-580-server-open, (kernel modules provided
by linux-modules-nvidia-580-server-open-generic)`. Pick the **highest branch
number that is 570 or newer**, preferring the **`-server-open`** variant.
NVIDIA recommends the open kernel modules for Hopper GPUs like the H200; the
plain `-server` variant also works.

Set the branch you picked. **580** is only an example; use the number from
your list:

```bash
BR=580
```

### 3.2 Install the driver and nvidia-smi

```bash
sudo ubuntu-drivers install --gpgpu nvidia:${BR}-server-open
sudo apt install -y nvidia-utils-${BR}-server
```

If the first command rejects `-server-open`, use
`sudo ubuntu-drivers install --gpgpu nvidia:${BR}-server` instead. With
`--gpgpu` the tool installs the compute driver only, so `nvidia-utils-...`
is a separate install: it provides `nvidia-smi`.

### 3.3 HGX (NVSwitch) systems only: Fabric Manager

Skip this on an H200 NVL / PCIe system.

```bash
sudo apt install -y nvidia-fabricmanager-${BR} libnvidia-nscq-${BR}
sudo systemctl enable nvidia-fabricmanager
```

The Fabric Manager branch must equal the driver branch. It pairs with the
`-server` driver even though its package name has no `-server`.

### 3.4 Reboot and check

```bash
sudo reboot
```

After the reboot:

```bash
nvidia-smi
```

**Check:** the table lists your H200 with a *Driver Version* of 570 or
higher, and no error. Then:

```bash
# keep the driver loaded between runs (faster start, stable clocks)
sudo systemctl enable --now nvidia-persistenced 2>/dev/null || sudo nvidia-smi -pm 1
nvidia-smi --query-gpu=name,persistence_mode,memory.total --format=csv

# HGX only: the fabric must be up, or CUDA fails with error 802
systemctl is-active nvidia-fabricmanager
nvidia-smi -q | grep -A 3 -i "fabric"          # State: Completed, Status: Success
```

If something is wrong, see [Troubleshooting](#troubleshooting) at the end.

---

## Part 4. Protect long runs from updates and sleep

Training takes days to weeks. Two things commonly break it.

**Automatic updates replacing the driver mid-run.** When the driver
libraries are updated but the loaded kernel module is not, `nvidia-smi` and
CUDA fail with *Driver/library version mismatch* until the next reboot. Pin
the driver and the kernel:

```bash
# every INSTALLED driver package and kernel meta-package (incl. Desktop's -hwe kernels)
PKGS=$(dpkg-query -W -f='${db:Status-Abbrev} ${Package}\n' \
  'nvidia-*' 'libnvidia-*' 'linux-modules-nvidia-*' \
  'linux-generic*' 'linux-image-generic*' 'linux-headers-generic*' 2>/dev/null | awk '$1=="ii" {print $2}')
sudo apt-mark hold $PKGS
apt-mark showhold
```

The kernel is pinned together with the driver on purpose. Upgrading the kernel
while the NVIDIA module package stays pinned leaves the new kernel without a
driver after the next reboot.

After your runs, release the pins with
`sudo apt-mark unhold $(apt-mark showhold)` and update normally.

**Ubuntu Desktop going to sleep.** Ubuntu Server never sleeps, but Desktop
suspends after inactivity and stops the run. Disable sleep on Desktop:

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

---

## Part 5. Prepare the data disk

The runs need **500 GB+** on one filesystem: the Python environment, models,
data, checkpoints and logs. See what you have:

```bash
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS
df -h
```

**Case A: a large filesystem is already mounted** (for example `/data`, or
`/` itself has 500 GB+ free). Just create the workspace:

```bash
sudo mkdir -p /data/workspace && sudo chown "$USER" /data/workspace
```

**Case B: a large disk is listed but has no mount point.** It needs a
filesystem.

> **This erases the disk you name.** Check `lsblk` twice. The disk must be
> the empty one, not the one holding `/`. In the example below it is
> `/dev/nvme1n1`; yours may differ.

```bash
DISK=/dev/nvme1n1                                   # <-- the EMPTY data disk from lsblk
sudo mkfs.ext4 -L opd-data "$DISK"
sudo mkdir -p /data
echo "LABEL=opd-data /data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
sudo systemctl daemon-reload && sudo mount -a
sudo mkdir -p /data/workspace && sudo chown "$USER" /data/workspace
df -h /data
```

**Check:** `df -h /data` shows the disk with 500 GB+ available. After a
reboot, it is still mounted.

---

## Part 6. Install everything the project needs

One script installs all of it into the workspace. It changes neither the
driver nor the system Python:

```bash
cd ~
git clone https://github.com/Amirgh8080/draft-opd-repro.git
cd draft-opd-repro
tmux new -s opd                                     # survives SSH disconnects
bash h200/setup_bare_metal.sh --workspace /data/workspace
```

This takes about 45–60 minutes. It asks before each `sudo` install; add `-y`
to accept them all.

| Installed | Where | Why |
|---|---|---|
| git, git-lfs, ffmpeg, libnuma-dev, build-essential, pkg-config, tmux | apt | ffmpeg: needed by torchcodec, an SGLang dependency; the rest: builds |
| CUDA toolkit 12.8 (`cuda-toolkit-12-8`, no driver) | `/usr/local/cuda-12.8`, from NVIDIA's apt repo | only to compile flash-attn for the H200 |
| uv | `~/.local/bin` | creates the Python environment |
| Python 3.12 | `/data/workspace/venv` | the version the pinned packages need |
| torch 2.9.1, SGLang DFlash fork, flashinfer, sgl-kernel, verl, Ray, transformers 4.57.1, … | `/data/workspace/venv` | the training and evaluation stack, pinned in `h200/constraints.txt` |
| flash-attn 2.8.3 (compiled for SM90) | `/data/workspace/venv` | the paper's attention path for the frozen target model |
| upstream Draft-OPD at the pinned commit | `/data/workspace/Draft-OPD` | the paper's code, with the launcher patch |

It ends with **"Bare-metal setup complete."** Then check:

```bash
bash h200/preflight.sh
```

**Check:** every line says `OK`, including the Python stack imports. If the
flash-attn compile runs out of memory, rerun the setup with fewer parallel
jobs: `MAX_JOBS=4 bash h200/setup_bare_metal.sh --workspace /data/workspace`.

**Optional: Docker.** You don't need Docker for this path. If you want it
for other work, follow [`GUIDE.md`](GUIDE.md) step A2 (Docker + NVIDIA
Container Toolkit).

---

## Part 7. Final checklist

| Check | Command | Expected |
|---|---|---|
| GPU and driver | `nvidia-smi` | H200 listed, driver ≥ 570 |
| Persistence | `nvidia-smi --query-gpu=persistence_mode --format=csv,noheader` | `Enabled` |
| Fabric (HGX only) | `nvidia-smi -q \| grep -A 3 -i fabric` | `Completed` / `Success` |
| Driver pinned | `apt-mark showhold \| grep -c nvidia` | more than 0 |
| Data disk | `df -h /data` | 500 GB+ available, survives reboot |
| CUDA toolkit | `/usr/local/cuda-12.8/bin/nvcc --version` | release 12.8 |
| Project setup | `cat ~/draft-opd-repro/h200/.local.env` | workspace, venv and CUDA paths |
| Preflight | `bash ~/draft-opd-repro/h200/preflight.sh` | all `OK` |

All green: the machine is ready. Continue with
[`GUIDE_LINUX.md` Step 6](GUIDE_LINUX.md#step-6-phase-1-reproduce-table-1-with-the-released-checkpoints)
to reproduce Table 1, then train.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `lspci` shows no NVIDIA device | Hardware level: card seating, power cables, or the slot is disabled in firmware. |
| `nvidia-smi`: *No devices were found* | Usually firmware: enable *Above 4G Decoding* (Part 0). Look at `sudo dmesg \| grep -i -E "nvrm\|nvidia"` for BAR or memory-mapping errors. |
| `nvidia-smi`: *couldn't communicate with the NVIDIA driver* | The module did not load. Run `lsmod \| grep nvidia`. If it is empty, check `sudo dmesg \| grep -i nvidia`. *Required key not available* means Secure Boot blocked an unsigned module: reinstall through `ubuntu-drivers` (Part 3), which uses signed modules, rather than a DKMS or `.run` driver. |
| *Driver/library version mismatch* | Driver packages were updated while the old module was loaded. Reboot, then pin the driver (Part 4). |
| CUDA *error 802: system not yet initialized* | HGX system without a working Fabric Manager. Install it at the **same branch** as the driver (Part 3.3), then `sudo systemctl restart nvidia-fabricmanager` and `journalctl -u nvidia-fabricmanager -b` to see why it failed. |
| `ubuntu-drivers list --gpgpu` shows nothing | Run `sudo apt update`, and check that `restricted` and `multiverse` are enabled in *Software & Updates* (Desktop) or `/etc/apt/sources.list.d/ubuntu.sources` (24.04). |
| Black screen after installing the driver (Desktop) | The monitor is on the H200, which has no display. Use the motherboard's video port or the BMC console. |
| `nouveau` conflicts | The driver packages blacklist it; a reboot after Part 3 is required. `lsmod \| grep nouveau` should print nothing. |
| Setup or preflight errors | See [`GUIDE_LINUX.md` → Troubleshooting](GUIDE_LINUX.md#troubleshooting). |

Sources for the driver steps: Ubuntu Server's
[NVIDIA drivers installation](https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers)
guide (ubuntu-drivers, `-server` branches, Fabric Manager pairing, Secure Boot)
and NVIDIA's
[Fabric Manager user guide](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html).
