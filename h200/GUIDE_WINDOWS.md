# Running from Windows

The Draft-OPD stack runs only on Linux. SGLang, sgl-kernel and flashinfer
publish no Windows builds, so nothing here runs in native Windows, PowerShell
or Python for Windows. From a Windows computer there are two ways to work:

| Your situation | Route | Status |
|---|---|---|
| **The H200 is in the cloud** (a VM or pod) and Windows is your laptop or desktop | **Route 1:** control the Linux machine from Windows over SSH | Recommended. This is the setup the scripts target. |
| **The H200 is physically in a Windows machine** (a workstation or Windows Server) | **Route 2:** WSL2 + Docker on that machine | Possible, with caveats (see [Route 2](#route-2-the-h200-is-in-a-windows-machine-wsl2)) |

In this guide every command block says where it runs:

- **PowerShell**: on Windows.
- **Ubuntu (WSL)**: inside WSL on Windows.
- **Remote**: on the Linux H200 machine, inside your SSH session.

Pasting a Linux command into PowerShell is the most common mistake, so check
the label before you paste.

---

## Route 1: Windows controls a Linux H200 in the cloud

Windows only opens a terminal on the remote machine. All the work happens on
Linux, by following [`GUIDE.md`](GUIDE.md).

### 1. Get a terminal and SSH

Windows 10 and 11 ship an OpenSSH client. Use **Windows Terminal**, which is
preinstalled on Windows 11 and available from the Microsoft Store on 10.

**PowerShell**
```powershell
ssh -V                                   # should print OpenSSH_for_Windows_...
```

If it is missing, open **Settings → System → Optional features → Add a
feature → OpenSSH Client**.

### 2. Create an SSH key and give it to the provider

**PowerShell**
```powershell
ssh-keygen -t ed25519                    # press Enter to accept the default path
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub | Set-Clipboard
```

Paste the copied public key into your provider's **SSH keys** settings, before
you create the machine or pod.

### 3. Keep SSH sessions from dropping

Idle SSH sessions get cut by routers and VPNs. Create or edit
`%USERPROFILE%\.ssh\config`:

**PowerShell**
```powershell
notepad $env:USERPROFILE\.ssh\config
```

Put this in the file. The host name `h200` is your choice; `HostName`, `User`
and `Port` come from your provider:

```
Host h200
    HostName 203.0.113.10
    User ubuntu
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 10
```

Save it **without** a `.txt` extension. In Notepad's Save dialog, choose
*All files*. You can now connect with:

**PowerShell**
```powershell
ssh h200
```

### 4. Follow the Linux guide on the remote machine

Inside the SSH session you are on Linux. Follow [`GUIDE.md`](GUIDE.md) from
the start: **Path A** for a VM, **Path B** for a pod.

Two habits matter more when you connect from a laptop:

- **Run everything inside `tmux` on the remote machine.** If Windows sleeps,
  Wi-Fi drops or you close the lid, the run keeps going. Reattach with
  `tmux attach -t opd`.

  **Remote**
  ```bash
  sudo apt-get install -y tmux     # pods run as root: drop "sudo"
  tmux new -s opd
  ```

  On Path A, the long runs in GUIDE.md are already detached Docker containers,
  so they survive a disconnect even without tmux. Use tmux anyway for the
  build in step A5, which takes 30–45 minutes.
- **Clone the repo on the remote machine, not on Windows.** If you clone with
  Git for Windows and upload the files, the shell scripts can end up with
  Windows (CRLF) line endings and fail with
  `/usr/bin/env: 'bash\r': No such file or directory`. If that happens:

  **Remote**
  ```bash
  sed -i 's/\r$//' h200/*.sh h200/*.py repro/*.sh repro/*.py
  ```

### 5. View results and logs from Windows (optional)

The simplest way is **VS Code** with the **Remote - SSH** extension. Connect to
`h200` and open `/workspace/logs` (or `/data/workspace/logs` on Path A).
`summary.md` renders as a table, and you can tail `train.log` in VS Code's
terminal.

### 6. Copy results to Windows

`scp` ships with Windows. This command copies the whole `logs` folder into
your Downloads:

**PowerShell**
```powershell
scp -r h200:/data/workspace/logs "$env:USERPROFILE\Downloads\draft-opd-results"     # Path A
scp -r h200:/workspace/logs      "$env:USERPROFILE\Downloads\draft-opd-results"     # Path B
```

To copy one extracted drafter (~1 GB), replace `N` with the step number.
`ls` the checkpoints folder on the remote machine to find it:

**PowerShell**
```powershell
scp -r h200:/workspace/checkpoints/qwen3-4b-ins-draftopd-h200x1/global_step_N/draft_model `
       "$env:USERPROFILE\Downloads\draft_model_N"
```

Do not copy whole checkpoint folders. Each one is ~25–30 GB. If you have WSL
installed (see Route 2, step 2), you can also use the selective `rsync` command
from GUIDE.md Step 9 inside an Ubuntu window.

---

## Route 2: the H200 is in a Windows machine (WSL2)

Read this first. Route 2 runs Linux inside a lightweight VM (WSL2), and the
GPU reaches it through Windows' GPU paravirtualisation:

- **Datacenter GPUs and WSL.** WSL cannot see a GPU whose Windows driver runs
  in **TCC** mode, and datacenter cards (H100, H200, H200 NVL) ship in TCC.
  These cards have no display engine, so they cannot use **WDDM**, the mode
  for desktop cards. Their route into WSL2 is **MCDM**, a compute-only driver
  model. MCDM needs a recent driver (R570+) and a recent Windows: the latest
  Windows 11 or Windows Server 2025. Step 1 switches the card. If MCDM is
  refused on your machine, Route 2 is not possible there; install Linux on it,
  or use Route 1.

  **For H100/H200 specifically, prefer installing Linux on the machine.**
  NVIDIA's data center driver release notes (R550, R570) list Windows support
  for the H200 only in shared NVSwitch virtualization configurations. A
  standalone Windows box with an H200 is outside that, and on at least one H200
  NVL the driver failed to start after the MCDM switch. Recovery is below
  under "If the driver does not start after the switch".
- **Speedups.** τ (acceptance length) is a property of the models and prompts,
  so it is still comparable to the paper. The speedups go through the
  virtualisation layer, so they may not match the paper's bare-metal Linux
  numbers.
- **Training is unverified on WSL2.** Training puts the SGLang rollout engine
  to sleep and wakes it on the same GPU, using CUDA virtual-memory calls, and
  pins large amounts of host memory. Both have known limitations under WSL.
  Evaluation (Phase 1) is the safer use of this route. If training fails in
  ways that do not reproduce on Linux, run Phase 2 on a Linux machine. The
  checkpoints are portable.

### 1. Check the Windows version, the driver and its mode

**Check the Windows version first.** For a datacenter card, Route 2 needs the
latest Windows 11 or Windows Server 2025. On **Windows 10** (any version) or
Windows Server 2022, switching to MCDM leaves the driver unable to start: the
card shows *Code 43* and `nvidia-smi` stops working. Do not switch on those
versions. Use Route 1, or install Linux on the machine.

**PowerShell**
```powershell
Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' | Select ProductName, DisplayVersion, CurrentBuild
```

A build number of 26100 or higher means Windows 11 24H2+ or Server 2025.
19045 is Windows 10 22H2. (`ProductName` can still say "Windows 10" on
Windows 11, so go by the build number.)

**PowerShell** (as Administrator)
```powershell
nvidia-smi
nvidia-smi -q | Select-String "Driver Model" -Context 0,2
```

`Current` must say **MCDM** (datacenter cards) or **WDDM** (desktop cards).
If it says **TCC**, switch a datacenter card to MCDM. Run PowerShell **as
Administrator** for this: right-click Windows Terminal → *Run as
administrator*. Without admin rights, the switch fails.

**PowerShell** (as Administrator)
```powershell
nvidia-smi -i 0 -dm 2          # 2 = MCDM. Do not use 0 (WDDM) on a datacenter card.
nvidia-smi -q | Select-String "Driver Model" -Context 0,2
```

Read what the first command prints. **Before you reboot, `Pending` must say
MCDM.** If it still says TCC, the switch was refused, and rebooting will not
help; the printed message says why. Once `Pending` says MCDM:

**PowerShell** (as Administrator)
```powershell
Restart-Computer
```

After the reboot, `Current` should say MCDM. Two problems are known:

- On Windows Server 2022, the card can show a yellow warning in Device Manager
  after the switch. Windows Server 2025 or the latest Windows 11 is the
  reported fix.
- To undo the switch, run `nvidia-smi -i 0 -dm 1` (TCC) and reboot. This only
  works while `nvidia-smi` can still reach the driver; see the next section if
  it cannot.

#### If the driver does not start after the switch

Symptom: after the reboot, every `nvidia-smi` command prints *"couldn't
communicate with the NVIDIA driver"*. The driver failed to load in MCDM mode,
so `nvidia-smi` can no longer switch the card back. Recover like this:

1. Check the device state:

   **PowerShell** (as Administrator)
   ```powershell
   Get-PnpDevice | Where-Object FriendlyName -like "*NVIDIA*" |
     ForEach-Object { [pscustomobject]@{
       Status = $_.Status; Class = $_.Class; Name = $_.FriendlyName
       Problem = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_Device_ProblemCode).Data } } |
     Format-Table -AutoSize
   ```

   A `Problem` code (for example 43, 10 or 31) confirms that the driver
   refused to start.
2. **Download the driver version that worked before**, the one `nvidia-smi`
   showed while the card was in TCC. Get it from nvidia.com/drivers (*Data
   Center / Tesla* → *H-Series* → *H200 NVL* → your Windows version). The
   same version is the safe choice: newer branches may drop your Windows
   version (R595 lists only Windows 11 24H2/25H2). Download it *before* step 3.
3. **Remove the device together with its driver.** The driver-model setting
   is stored with the installed device, so deleting the device is what resets
   it. Open **Device Manager** (`devmgmt.msc`) and find *NVIDIA H200 NVL*, which
   has a yellow warning sign. It may be under *Display adapters*, *Compute
   accelerators* or *Other devices*. Right-click it → *Uninstall device*, tick
   *Delete the driver software for this device* (Windows 11: *Attempt to
   remove the driver for this device*), then *Uninstall*. **Reboot.**
4. **Install the driver from step 2**: *Custom (Advanced)* → tick *Perform a
   clean installation*, then reboot.
5. **Check:**

   **PowerShell** (as Administrator)
   ```powershell
   nvidia-smi
   nvidia-smi -q | Select-String "Driver Model" -Context 0,2
   ```

   It should list the card again, with `Current: TCC`. If it still shows
   Code 43 or no card, repeat step 3. If Windows reinstalled a driver on its
   own after the reboot in step 3, delete the device and driver once more, so
   the clean install in step 4 starts from scratch.

On Windows 10, Route 2 ends here. Use Route 1, install Ubuntu on the machine
and follow [`GUIDE.md`](GUIDE.md) Path A, or upgrade to Windows 11 24H2+ and
try again with a current driver. That last option is still unofficial for the
H200 (see above).

If MCDM is refused on your Windows version, stop here: install Linux on the
machine, or use Route 1.

Use a recent NVIDIA Windows driver: R570 or newer for MCDM (R575+ also avoids
the `NVIDIA_DISABLE_REQUIRE` workaround). **Never install a Linux NVIDIA driver
inside WSL.** The Windows driver provides the GPU to WSL.

### 2. Install WSL2 with Ubuntu

**PowerShell** (as Administrator)
```powershell
wsl --install -d Ubuntu-24.04
wsl --update
```

Reboot if asked. Then open **Ubuntu 24.04** from the Start menu and create a
Linux user name and password when prompted.

### 3. Give WSL enough memory and put it on the big drive

By default WSL gets only half of the machine's RAM, and its virtual disk lives
on `C:`.

Create `%USERPROFILE%\.wslconfig`:

**PowerShell**
```powershell
notepad $env:USERPROFILE\.wslconfig
```

Put this in the file and adjust it to your machine. The recipe wants 120 GB+
of RAM and 16+ CPUs:

```ini
[wsl2]
memory=200GB
processors=32
swap=32GB
```

Then move the Ubuntu disk to a drive with 500 GB+ free. `D:\WSL` is an example
path:

**PowerShell**
```powershell
wsl --shutdown
wsl --manage Ubuntu-24.04 --move D:\WSL
```

On older WSL versions without `--manage`, use
`wsl --export Ubuntu-24.04 D:\ubuntu.tar` and then
`wsl --import Ubuntu-24.04 D:\WSL D:\ubuntu.tar`. Before importing, remove
the old copy with `wsl --unregister Ubuntu-24.04`. That command **deletes**
the distro, so only run it after the export has finished.

Check the new settings:

**Ubuntu (WSL)**
```bash
free -g          # total should match memory= above
nproc
df -h /          # should show the big drive's space
```

### 4. Confirm the GPU is visible inside WSL

**Ubuntu (WSL)**
```bash
nvidia-smi
```

It should show the H200. If it says `command not found` or `No devices were
found`, go back to step 1. The usual cause is the driver model (still TCC) or an
outdated Windows driver.

### 5. Install Docker inside WSL

Use Docker Engine inside Ubuntu, so the commands in GUIDE.md work unchanged.
Docker Desktop with the WSL2 backend also works, but install only one of the
two.

**Ubuntu (WSL)**
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
```

Then install the NVIDIA Container Toolkit: run the "NVIDIA Container Toolkit"
block from **GUIDE.md step A2** in this Ubuntu window. Close Ubuntu, run
`wsl --shutdown` in PowerShell, and reopen Ubuntu so the group change takes
effect. If `docker ps` then says it cannot connect, start Docker with
`sudo service docker start`.

### 6. Keep everything on the Linux filesystem

Inside WSL, `C:` and `D:` appear as `/mnt/c` and `/mnt/d`. **Do not put the
repo or the workspace there.** That filesystem is many times slower, breaks
Linux file permissions, and triggers git's "dubious ownership" error during
the install.

Use a path inside Ubuntu instead:

**Ubuntu (WSL)**
```bash
sudo mkdir -p /data/workspace && sudo chown "$USER" /data/workspace
cd ~ && git clone https://github.com/Amirgh8080/draft-opd-repro.git
```

Because `/data` is inside the Ubuntu disk you moved in step 3, it already lives
on the big drive. Skip GUIDE.md's step A3 entirely; the Docker data-root move
is not needed here.

To open results from Windows, type this in File Explorer's address bar:
`\\wsl$\Ubuntu-24.04\data\workspace\logs`

### 7. Keep Windows awake for the whole run

A sleeping or rebooting Windows machine stops WSL, and the run with it.

**PowerShell** (as Administrator)
```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

Also open **Settings → Windows Update** and pause updates for the duration of
the run, or set active hours, so Windows does not restart overnight.

Keep one **Ubuntu** window open during the run. WSL can shut its VM down when
no window is attached, and Docker containers stop with it. A `tmux` session
inside that window lets you close and reattach other windows freely.

### 8. Follow the Linux guide inside Ubuntu

From here, follow [`GUIDE.md`](GUIDE.md) **Path A** inside the Ubuntu window,
with these adjustments:

| GUIDE.md step | In WSL |
|---|---|
| A1 Check driver | done in step 1 above; `nvidia-smi` works inside Ubuntu |
| A2 Docker + toolkit | done in step 5 above |
| A3 Big disk | **skip**: step 3 moved the disk and step 6 created `/data/workspace` |
| A4 → A7 | as written (`cd ~/draft-opd-repro` before A5) |
| Steps 5–9 | as written |

`preflight.sh` detects WSL and warns that the speedups may not be comparable
to the paper. It also flags a workspace that sits on a Windows drive.

---

## Troubleshooting on Windows

| Symptom | Cause and fix |
|---|---|
| `bash\r: No such file or directory` or `$'\r': command not found` | CRLF line endings: the repo was cloned or edited on Windows. Run the `sed -i 's/\r$//'` command from Route 1, step 4, or re-clone inside Linux/WSL. |
| `ssh: connect to host ... Connection timed out` | Wrong `HostName` or `Port` in `.ssh\config`, or the provider's firewall blocks it. Pods often use a non-22 port. |
| `Permission denied (publickey)` | The key was not added to the provider before the machine was created, or `IdentityFile` points to the wrong file. |
| WSL `nvidia-smi`: `No devices were found` | GPU still in TCC mode, or outdated Windows driver: switch to MCDM (Route 2, step 1). |
| After switching to MCDM and rebooting, `nvidia-smi` says it *couldn't communicate with the NVIDIA driver*; Device Manager shows Code 43 | The driver cannot start in MCDM, which always happens on Windows 10 and Server 2022. Delete the device with its driver, then reinstall the previously working driver (Route 2, step 1, "If the driver does not start after the switch"). |
| `nvidia-smi -dm 0` changes nothing; `Pending` stays TCC | Datacenter cards cannot use WDDM. Use `-dm 2` (MCDM) in an Administrator PowerShell (Route 2, step 1). |
| `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit not installed or not configured: redo GUIDE.md A2's toolkit block, then `sudo service docker restart`. |
| `unsatisfied condition: cuda>=12.9` | Windows driver older than R575: add `-e NVIDIA_DISABLE_REQUIRE=1` to `OPD_FLAGS` (GUIDE.md A6), or update the driver. |
| The run stops overnight | Windows slept or installed updates (Route 2, step 7), or the Ubuntu window was closed. On Route 1 this does not happen: the run lives on the remote machine. |
| Install is very slow, or git reports `dubious ownership` | The repo or workspace is under `/mnt/c` or `/mnt/d`. Move it into the Linux filesystem (Route 2, step 6). |
| `free -g` in WSL shows half the RAM | `.wslconfig` was saved as `.wslconfig.txt`, or WSL was not restarted: run `wsl --shutdown`. |
