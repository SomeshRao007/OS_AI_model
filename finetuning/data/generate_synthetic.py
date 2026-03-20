"""
Generate synthetic training data using Claude API (Haiku).

Generates 3 types of task-based training examples:
  1. Direct-answer (~13,600): question -> one command + one-line explanation
  2. Clarification (~4,400): ambiguous request -> ask one clarifying question
  3. Troubleshooting (~5,000): "X is broken" -> one diagnostic/fix command
  Total: ~23,000 high-quality task-based examples

Requires: ANTHROPIC_API_KEY env var, anthropic Python package
Estimated cost: ~$7.00-8.00

Usage:
  python generate_synthetic.py                      # generate all
  python generate_synthetic.py --direct-only         # only direct-answer examples
  python generate_synthetic.py --clarification-only  # only clarification examples
  python generate_synthetic.py --troubleshoot-only   # only troubleshooting examples
  python generate_synthetic.py --dry-run             # show prompts without API calls
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"

MODEL = "claude-haiku-4-5-20251001"
BATCH_SIZE = 50

NEW_SYSTEM_PROMPT = (
    "You are an AI assistant built into a Linux-based operating system. "
    "When asked to perform a task, respond with one correct command followed "
    "by a brief one-line explanation. If the request is ambiguous, ask one "
    "clarifying question instead of guessing. For conceptual questions, give "
    "a clear and focused explanation. Never list multiple alternative commands. "
    "Never restate the question."
)

# ---------------------------------------------------------------------------
# Domain definitions — 17 domains × 800 = 13,600 direct-answer examples
# Each domain has subtopics to ensure diversity across batches
# ---------------------------------------------------------------------------

DIRECT_DOMAINS = {
    "file_operations": {
        "count": 800,
        "commands": (
            "find, chmod, chown, chgrp, ln, stat, file, locate, updatedb, cp, mv, rm, "
            "mkdir, rmdir, touch, tar, zip, unzip, gzip, gunzip, xz, unxz, bzip2, bunzip2, "
            "rsync, readlink, basename, dirname, realpath, rename, install, mktemp, "
            "shred, truncate, split, csplit, tree, ncdu, pathchk, sync, tee"
        ),
        "description": "File/directory operations, permissions, links, archiving, batch file manipulation",
        "subtopics": [
            "finding and locating files with find (by name, size, time, type, permissions, exec actions), locate, which, whereis",
            "file permissions and ownership with chmod (numeric 755 and symbolic u+x), chown, chgrp recursively",
            "archiving and compression with tar (create, extract, list, specific files), zip, gzip, xz, bzip2, split archives",
            "copying, moving, syncing files with cp (-a, -r, -u), mv, rsync (--progress, --exclude, --delete)",
            "file metadata and info with stat (format strings), file (MIME types), ls -la, du (-sh, --max-depth)",
            "symbolic and hard links with ln (-s, -f), readlink (-f), realpath, managing broken symlinks",
            "creating, removing, organizing files: touch, mkdir -p, rm -rf, rmdir, mktemp, shred, truncate, tree",
        ],
    },
    "networking": {
        "count": 800,
        "commands": (
            "ss, netstat, ip addr, ip route, ip link, ip neigh, curl, wget, ssh, scp, rsync, "
            "iptables, nft, nftables, ping, ping6, traceroute, tracepath, dig, nslookup, nc, "
            "netcat, tcpdump, hostname, hostnamectl, ifconfig, nmap, route, arp, arping, "
            "ethtool, mtr, whois, host, resolvectl, nmcli, iw, iwconfig, bridge, socat, "
            "ssh-keygen, ssh-copy-id, ssh-agent, ssh-add, sshfs, wget2, aria2c"
        ),
        "description": "Network diagnostics, file transfers, firewalls, DNS, SSH, WiFi, routing",
        "subtopics": [
            "network diagnostics with ping (count, interval, size), traceroute, mtr, ss (-tlnp, -tunap), netstat",
            "DNS lookups and resolution with dig (+short, +trace, MX, AAAA), nslookup, host, whois, resolvectl",
            "file transfers with scp (-r, -P), rsync (--progress, --compress), wget (-c, -r), curl (-o, -L, -H, -d)",
            "SSH connections, tunnels (-L, -R, -D), key management (ssh-keygen, ssh-copy-id, ssh-agent), sshfs, config",
            "firewall rules with iptables (-A, -D, -L, -F, ACCEPT/DROP/REJECT, port forwarding) and nftables",
            "network interface config with ip addr (add/del), ip link (up/down), ip route, nmcli, ethtool, bridge",
            "packet capture and analysis with tcpdump (-i, -w, -r, filters, port, host), nc/netcat, socat, nmap (-sV, -sS)",
        ],
    },
    "process_management": {
        "count": 800,
        "commands": (
            "ps, kill, killall, pkill, pgrep, top, htop, systemctl, journalctl, "
            "crontab, nohup, screen, tmux, bg, fg, jobs, nice, renice, ionice, "
            "strace, ltrace, lsof, wait, disown, at, atq, atrm, batch, watch, "
            "timeout, xargs, parallel, flock, pstree, pidof, taskset, chrt, cpulimit"
        ),
        "description": "Process control, systemd services, scheduling, multiplexing, debugging, resource limits",
        "subtopics": [
            "listing and filtering processes with ps (aux, -ef, -o custom), pgrep (-f, -l), top (-bn1), htop, pstree",
            "killing and signaling processes with kill (-9, -15, -HUP), killall, pkill (-f), timeout, cpulimit",
            "systemd service management with systemctl (start/stop/restart/enable/disable/status/mask), journalctl (-u, -f, --since)",
            "scheduling tasks with crontab (-e, -l, -r, syntax), at, atq, atrm, systemd timers",
            "background jobs with nohup, bg, fg, jobs, disown, screen (sessions, detach, reattach), tmux (sessions, panes, windows)",
            "process priority and affinity with nice (-n), renice, ionice (-c), taskset, chrt",
            "process debugging with strace (-p, -e, -f), ltrace, lsof (-i, -p, +D), flock, watch (-n, -d)",
        ],
    },
    "user_permissions": {
        "count": 800,
        "commands": (
            "useradd, usermod, userdel, passwd, groups, groupadd, groupmod, groupdel, "
            "sudoers, visudo, chage, id, who, w, last, lastb, lastlog, setfacl, getfacl, "
            "su, sudo, newgrp, gpasswd, finger, chsh, chfn, adduser, deluser, addgroup, "
            "login.defs, /etc/passwd, /etc/shadow, /etc/group, /etc/gshadow, nologin, "
            "faillock, pam_tally2, pwck, grpck, vipw, vigr"
        ),
        "description": "User/group management, sudo, password policies, ACLs, PAM, account security",
        "subtopics": [
            "creating and managing users with useradd (-m, -s, -G, -d, -e), usermod (-aG, -L, -U, -s), userdel (-r)",
            "password policies and aging with passwd (-l, -u, -e, -x, -n), chage (-l, -M, -m, -W, -E, -I)",
            "group management with groupadd, groupmod, groupdel, groups, newgrp, gpasswd (-a, -d, -A), /etc/group",
            "sudo configuration with visudo, sudoers syntax (NOPASSWD, Cmnd_Alias, User_Alias, Defaults), sudo -l",
            "access control lists with setfacl (-m, -x, -b, -R, default), getfacl, ACL inheritance",
            "user info and login tracking with id, who (-a), w, last (-n), lastb, lastlog, faillock",
            "switching users and shells with su (-, -c), sudo (-u, -i, -s, -E), chsh, /etc/login.defs, nologin",
        ],
    },
    "package_management": {
        "count": 800,
        "commands": (
            "apt, apt-get, apt-cache, apt-mark, apt-file, dpkg, dpkg-query, dpkg-reconfigure, "
            "snap, pip, pip3, venv, virtualenv, pipx, dnf, yum, pacman, paru, yay, flatpak, "
            "rpm, rpmquery, gem, npm, npx, yarn, cargo, brew, nix, add-apt-repository, "
            "update-alternatives, apt-key, sources.list, gpg key verification"
        ),
        "description": "Package install/remove/search/update across distros, language managers, repos",
        "subtopics": [
            "Debian/Ubuntu with apt (install, remove, purge, autoremove, update, upgrade, full-upgrade), apt-cache (search, show, depends), dpkg (-i, -r, -l, -S, -L)",
            "RHEL/Fedora with dnf (install, remove, update, search, info, group install, history, rollback), yum, rpm (-ivh, -qa, -qf, -e)",
            "Arch Linux with pacman (-S, -Ss, -R, -Rns, -Syu, -Q, -Ql, -Qo), paru, yay for AUR",
            "Python packages with pip install (--user, -r requirements.txt, --upgrade), pip freeze, venv, virtualenv, pipx",
            "Node.js/Rust/Ruby with npm (install -g, -D, ci, audit, ls), npx, yarn, cargo (install, build, update), gem",
            "pinning, holding, and managing versions: apt-mark (hold, unhold, showhold), dnf versionlock, pip constraints",
            "adding repos, PPAs, keys: add-apt-repository, sources.list.d, apt-key, gpg --recv-keys, dnf config-manager, flatpak remote-add",
        ],
    },
    "text_processing": {
        "count": 800,
        "commands": (
            "grep, egrep, fgrep, grep -P (PCRE), sed, awk, gawk, sort, uniq, cut, tr, wc, "
            "head, tail, jq, yq, diff, colordiff, comm, paste, tee, xargs, column, fmt, "
            "fold, nl, rev, expand, unexpand, strings, od, hexdump, xxd, iconv, dos2unix, "
            "unix2dos, envsubst, printf, seq, shuf, yes, base64"
        ),
        "description": "Text search, transformation, filtering, JSON/YAML processing, encoding",
        "subtopics": [
            "pattern matching with grep (-r, -i, -l, -c, -n, -v, -E, -P, -o, -A/-B/-C context), egrep, fgrep",
            "stream editing with sed (s///g, -i, delete lines, insert, append, ranges, hold space, backreferences)",
            "field processing with awk (print $1, -F delimiter, patterns, BEGIN/END, arrays, built-in variables, printf)",
            "sorting, dedup, set ops: sort (-n, -k, -t, -u, -r, -h, -V), uniq (-c, -d, -u), comm (-12, -13, -23)",
            "extracting fields: cut (-d, -f, -c), tr (-d, -s, -c, character classes), column (-t, -s), paste (-d)",
            "JSON/YAML processing: jq (., .key, .[], select(), map(), keys, @csv, -r), yq, base64 encode/decode",
            "combining tools: pipes, xargs (-I, -P, -0, -n), tee (-a), process substitution, complex text pipelines",
        ],
    },
    "storage_filesystems": {
        "count": 800,
        "commands": (
            "mount, umount, df, du, lsblk, fdisk, gdisk, parted, mkfs, mkfs.ext4, "
            "mkfs.xfs, mkfs.btrfs, blkid, fstab, /etc/fstab, dd, lvs, pvs, vgs, "
            "lvcreate, lvextend, lvreduce, lvremove, pvcreate, pvremove, vgcreate, "
            "vgextend, mdadm, e2fsck, fsck, tune2fs, resize2fs, xfs_growfs, xfs_repair, "
            "swapon, swapoff, mkswap, fallocate, findmnt, mountpoint, btrfs, zfs, "
            "smartctl, hdparm, badblocks, partprobe, wipefs"
        ),
        "description": "Disk management, partitioning, filesystems, LVM, RAID, SMART, swap",
        "subtopics": [
            "checking disk space with df (-h, -i, -T), du (-sh, --max-depth, --exclude, -a), ncdu, lsblk (-f, -o)",
            "mounting/unmounting: mount (-t, -o, bind, loop, remount), umount (-l), /etc/fstab syntax, findmnt, mountpoint",
            "partitioning with fdisk, gdisk (GPT), parted (mklabel, mkpart, resizepart, print), partprobe, wipefs",
            "creating filesystems: mkfs.ext4, mkfs.xfs, mkfs.btrfs, tune2fs (-l, -c, -i, -L), blkid",
            "LVM management: pvcreate, vgcreate, lvcreate (-L, -n, -l), lvextend (-L, -r), lvreduce, lvs, pvs, vgs, snapshots",
            "RAID setup with mdadm (--create, --detail, --examine, --fail, --remove, --add, /etc/mdadm/mdadm.conf)",
            "disk health, swap, imaging: smartctl (-a, -t), dd (if=, of=, bs=, status=progress), swapon, swapoff, mkswap, fallocate, fsck",
        ],
    },
    "shell_scripting": {
        "count": 800,
        "commands": (
            "for loops, while read loops, until loops, if/elif/else, [[ ]] vs [ ], "
            "case/esac, select, functions, local, return, getopts, $1/$@/$#/$?/$$, "
            "set -euo pipefail, set -x, trap (EXIT, ERR, INT, TERM), pipes, "
            "subshells $(), command substitution, variable expansion ${var:-default}, "
            "${var:+alt}, ${var#pattern}, ${#var}, read (-p, -r, -a, -t), arrays, "
            "associative arrays, here-documents <<EOF, here-strings <<<, "
            "process substitution <(), exec, eval, shift, source, mapfile/readarray"
        ),
        "description": "Bash scripting: control flow, functions, error handling, string manipulation, arrays",
        "subtopics": [
            "for loops (C-style, list, glob, find -exec vs while read), while loops, until loops, break, continue",
            "conditionals: if/elif/else, [[ ]] vs [ ], test operators (-f, -d, -z, -n, -eq, -gt, =~), && and ||",
            "case/esac for pattern matching, select for menus, combining with functions",
            "functions: declaration, local variables, return codes, passing arrays, recursive functions",
            "argument parsing: positional params ($1, $@, $#), shift, getopts with optstring, OPTARG, OPTIND",
            "error handling: set -euo pipefail, trap (cleanup on EXIT, ERR), exit codes, $?, error messages to stderr",
            "string/variable manipulation: ${var:-default}, ${var:=value}, ${var#prefix}, ${var%suffix}, ${var//find/replace}, ${#var}, arrays, mapfile",
        ],
    },
    "kernel_os_concepts": {
        "count": 800,
        "commands": (
            "modprobe, lsmod, rmmod, insmod, modinfo, depmod, dmesg, sysctl, "
            "uname (-a, -r, -m), uptime, free (-h, -m), vmstat, iostat, sar, "
            "/proc/cpuinfo, /proc/meminfo, /proc/mounts, /proc/net, /proc/sys, "
            "/sys/class, /sys/block, cgroups, cgcreate, cgset, namespaces, "
            "unshare, nsenter, signals (SIGTERM, SIGKILL, SIGHUP, SIGUSR1), "
            "ulimit (-n, -u, -f, -a), nproc, lscpu, lspci, lsusb, lshw, "
            "dmidecode, hwinfo, inxi, numactl, taskset, chroot"
        ),
        "description": "Kernel modules, /proc & /sys, hardware info, cgroups, namespaces, signals, limits",
        "subtopics": [
            "kernel module management: modprobe (-r, --show-depends), lsmod, rmmod, modinfo, depmod, blacklisting",
            "system/hardware info: uname -a, lscpu, lspci (-v), lsusb, lshw (-short), dmidecode (-t), inxi",
            "memory and CPU monitoring: free -h, vmstat (procs, memory, io), iostat (-x), sar (-u, -r, -d), /proc/meminfo",
            "reading and writing /proc and /sys: /proc/cpuinfo, /proc/net/dev, /proc/sys/net, /sys/class, /sys/block",
            "resource limits: ulimit (-n files, -u procs, -f fsize, -a), /etc/security/limits.conf, cgroups v1/v2",
            "namespaces and isolation: unshare (--pid, --net, --mount), nsenter (-t, -n, -m), chroot, pivot_root",
            "kernel parameters and boot: sysctl (-w, -p, -a), /etc/sysctl.conf, dmesg (-T, -l, --follow), signals",
        ],
    },
    # --- NEW DOMAINS ---
    "docker_containers": {
        "count": 800,
        "commands": (
            "docker run, docker build, docker ps, docker images, docker exec, "
            "docker logs, docker stop, docker start, docker restart, docker rm, "
            "docker rmi, docker compose up, docker compose down, docker compose logs, "
            "docker compose ps, docker compose build, docker volume ls, docker volume create, "
            "docker volume rm, docker volume inspect, docker network ls, docker network create, "
            "docker network connect, docker network inspect, docker inspect, docker pull, "
            "docker tag, docker push, docker system prune, docker system df, docker cp, "
            "docker stats, docker top, docker diff, docker commit, docker save, docker load, "
            "docker export, docker import, docker history, docker port, "
            "Dockerfile: FROM, RUN, COPY, ADD, CMD, ENTRYPOINT, ENV, EXPOSE, VOLUME, WORKDIR, "
            "ARG, LABEL, HEALTHCHECK, USER, SHELL, .dockerignore, multi-stage builds"
        ),
        "description": "Docker container lifecycle, images, volumes, networking, compose, Dockerfiles",
        "subtopics": [
            "running containers: docker run (-d, -p, -v, -e, --name, --rm, --network, --restart, -it, --memory, --cpus)",
            "building images: docker build (-t, -f, --no-cache, --build-arg), Dockerfile best practices, multi-stage builds, .dockerignore",
            "managing containers: docker ps (-a, -q, --filter), stop, start, restart, rm (-f), exec (-it), logs (-f, --tail, --since)",
            "docker compose: up (-d, --build), down (-v, --rmi), logs (-f), ps, build, config, exec, scale",
            "docker volumes: create, ls, inspect, rm, prune, bind mounts vs named volumes, tmpfs mounts",
            "docker networking: network create (--driver bridge/overlay), connect, disconnect, inspect, port mapping, DNS resolution",
            "image management: images, pull, tag, push, rmi, save/load, export/import, history, system prune (-a, -f), system df",
        ],
    },
    "git_version_control": {
        "count": 800,
        "commands": (
            "git init, git clone (--depth, --branch, --single-branch), git add (-A, -p, -u), "
            "git commit (-m, --amend, --allow-empty), git push (-u, --force-with-lease, --tags), "
            "git pull (--rebase), git fetch (--all, --prune), git status (-s, -b), "
            "git log (--oneline, --graph, --all, --author, --since, --grep, -p, --stat), "
            "git diff (--staged, --name-only, --stat, HEAD~N), git branch (-a, -d, -D, -m, --merged), "
            "git checkout (-b, --, -f), git switch (-c), git merge (--no-ff, --squash, --abort), "
            "git rebase (-i, --onto, --abort, --continue), git stash (push, pop, list, drop, apply, show), "
            "git reset (--soft, --mixed, --hard, HEAD~N), git revert (-n, --no-commit), "
            "git cherry-pick (-x, --no-commit), git tag (-a, -d, -l), git remote (-v, add, remove, set-url), "
            "git blame (-L), git bisect (start, bad, good, reset), git reflog, "
            "git clean (-fd, -fdx, -n), git worktree, git submodule, git archive, "
            ".gitignore, .gitattributes, git config (--global, --local)"
        ),
        "description": "Git version control: workflow, branching, merging, rebasing, history, advanced ops",
        "subtopics": [
            "basic workflow: git add (specific files, -p patch mode), commit (-m, --amend), push (-u origin), pull (--rebase), status, clone",
            "branching: git branch (-a, -d, -D, --merged), checkout -b, switch -c, listing and cleaning merged branches",
            "merging and rebasing: merge (--no-ff, --squash, --abort), rebase (-i for squash/fixup/reorder, --onto, --abort, --continue)",
            "viewing history: git log (--oneline --graph --all, --author, --since, --grep, -p, --stat, -S pickaxe), diff (--staged, --stat)",
            "undoing changes: git reset (--soft/--mixed/--hard, HEAD~N), revert, checkout --, restore, stash (push -m, pop, list, drop)",
            "remote management: remote -v, add, remove, set-url, fetch --all --prune, push --force-with-lease, tracking branches",
            "advanced: cherry-pick, tag (-a annotated), blame (-L range), bisect (start/bad/good/reset), reflog, clean (-fd), worktree, submodule, archive, .gitignore patterns",
        ],
    },
    "system_monitoring": {
        "count": 800,
        "commands": (
            "top (-bn1, -o), htop, atop, iotop (-o), iftop, nload, nethogs, bmon, "
            "sar (-u, -r, -d, -n, -q, -b), mpstat (-P ALL), pidstat (-d, -r, -u, -p), "
            "vmstat (-s, -d, -w), iostat (-x, -p), dstat, glances, free (-h, -w, -s), "
            "uptime, w, who, tload, nmon, collectl, "
            "perf (stat, record, report, top), strace (-c, -e, -f, -p, -T), "
            "ltrace (-c, -e), time, /usr/bin/time -v, "
            "/proc/meminfo, /proc/loadavg, /proc/stat, /proc/diskstats, "
            "ss -s (summary), watch (-n, -d), slabtop"
        ),
        "description": "Real-time system monitoring, performance analysis, CPU/memory/disk/network bottlenecks",
        "subtopics": [
            "CPU monitoring: top (-bn1 -o %CPU), htop, mpstat -P ALL, pidstat -u, sar -u, perf top, load average interpretation",
            "memory analysis: free -h, vmstat (si/so columns), sar -r, /proc/meminfo, slabtop, smem, pmap -x",
            "disk I/O monitoring: iotop -o, iostat -x (await, %util), pidstat -d, sar -d, dstat, /proc/diskstats",
            "network traffic: iftop (-i), nload, nethogs, bmon, sar -n DEV, ss -s (summary statistics)",
            "historical data: sar (collected via sysstat/sadc), atop playback, collectl, dstat --output",
            "process-level profiling: strace (-c summary, -e trace=open, -T timestamps), ltrace, perf stat/record/report, time vs /usr/bin/time -v",
            "live dashboards: watch -n1 -d, htop customization, glances, nmon, tload, combining tools for custom monitoring",
        ],
    },
    "security_hardening": {
        "count": 800,
        "commands": (
            "ssh-keygen (-t ed25519, -b, -C, -f, -N), ssh-copy-id, ssh-agent, ssh-add, "
            "fail2ban-client (status, set, reload), ufw (allow, deny, delete, status, enable, disable, app list), "
            "firewall-cmd (--add-service, --add-port, --permanent, --reload, --list-all, --zone), "
            "selinux: getenforce, setenforce, sestatus, semanage, restorecon, audit2allow, "
            "apparmor: aa-status, aa-enforce, aa-complain, aa-logprof, "
            "chroot, openssl (genrsa, req, x509, s_client, verify, enc, dgst, rand), "
            "gpg (--gen-key, --encrypt, --decrypt, --sign, --verify, --import, --export, --list-keys), "
            "certbot (certonly, renew, --nginx, --apache, certificates, delete), "
            "aide (--init, --check, --update), auditd, auditctl (-w, -a, -l), ausearch, aureport, "
            "lynis (audit system), chkrootkit, rkhunter (--check, --update), "
            "passwd policies, PAM (/etc/pam.d/), umask, pwquality, "
            "chmod (setuid, setgid, sticky bit), chattr (+i, +a), lsattr"
        ),
        "description": "SSH security, firewalls, SELinux/AppArmor, certificates, encryption, auditing, hardening",
        "subtopics": [
            "SSH key management: ssh-keygen (-t ed25519, -b 4096), ssh-copy-id, authorized_keys, ssh-agent, ssh-add, ~/.ssh/config, known_hosts",
            "firewall with ufw: enable, allow/deny (port, from IP, to any port, app), status verbose, delete rules, logging, default policies",
            "firewall with firewall-cmd: --add-service, --add-port, --permanent, --reload, --list-all, zones, rich rules",
            "SSL/TLS with openssl: genrsa, req -new, x509, s_client -connect, verify chain, enc -aes-256-cbc, dgst -sha256, certbot",
            "file encryption and signing: gpg --gen-key, --encrypt -r, --decrypt, --sign, --verify, --armor, --export, keyservers",
            "SELinux (getenforce, setenforce, sestatus, semanage fcontext, restorecon, audit2allow) and AppArmor (aa-status, aa-enforce, profiles)",
            "auditing and scanning: auditctl -w (watch files), ausearch, aureport, aide --init/--check, lynis audit, chkrootkit, rkhunter, chattr +i, umask",
        ],
    },
    "log_analysis": {
        "count": 800,
        "commands": (
            "journalctl (-u, -f, --since, --until, -p, -b, --no-pager, -o json, --vacuum-size, --disk-usage), "
            "tail (-f, -n, -F), head, less, more, zless, zcat, zgrep, bzgrep, xzgrep, "
            "logrotate (-f, -d, /etc/logrotate.conf, /etc/logrotate.d/), logger (-t, -p, -s), "
            "syslog, rsyslog (/etc/rsyslog.conf, /etc/rsyslog.d/), syslog-ng, "
            "/var/log/syslog, /var/log/messages, /var/log/auth.log, /var/log/secure, "
            "/var/log/kern.log, /var/log/boot.log, /var/log/dpkg.log, /var/log/apt/, "
            "/var/log/nginx/, /var/log/apache2/, /var/log/mysql/, "
            "dmesg (-T, -l, --follow, -H), last, lastb, faillog, wtmp, btmp, "
            "grep, awk, sed for log parsing, multitail, lnav, ccze"
        ),
        "description": "Log viewing, systemd journal, rotation, syslog, real-time monitoring, parsing",
        "subtopics": [
            "systemd journal: journalctl -u service, -f follow, --since/--until, -p err, -b boot, -o json, --vacuum-size, --disk-usage",
            "real-time log monitoring: tail -f, tail -F (follow rename), multitail, journalctl -f, less +F, lnav",
            "compressed logs: zless, zgrep, zcat, bzgrep, xzgrep for rotated .gz/.bz2/.xz logs",
            "log rotation: logrotate config (daily/weekly, rotate N, compress, delaycompress, missingok, notifempty, postrotate, create)",
            "auth and security logs: /var/log/auth.log, /var/log/secure, last, lastb, faillog, wtmp analysis, failed SSH attempts",
            "parsing and extracting: grep patterns (timestamps, IPs, errors), awk field extraction, sed filtering, counting error frequencies, top talkers",
            "custom logging: logger -t tag -p facility.severity, rsyslog config (templates, filters, forwarding), syslog-ng, /etc/rsyslog.d/",
        ],
    },
    "environment_config": {
        "count": 800,
        "commands": (
            "env, export, printenv, set, unset, source, . (dot), .bashrc, .bash_profile, "
            ".profile, .bash_logout, /etc/environment, /etc/profile, /etc/profile.d/, "
            "/etc/bash.bashrc, alias, unalias, type, which, hash, "
            "PATH manipulation, LD_LIBRARY_PATH, MANPATH, XDG directories, "
            "locale, localectl (set-locale, list-locales), "
            "timedatectl (set-timezone, list-timezones, set-ntp, status), "
            "hostnamectl (set-hostname, status), "
            "update-alternatives (--config, --install, --display, --list), "
            "systemd-tmpfiles, tmpfiles.d, /etc/default/, /etc/sysconfig/, "
            "dircolors, inputrc, readline"
        ),
        "description": "Environment variables, shell config, locale, timezone, hostname, alternatives, PATH",
        "subtopics": [
            "environment variables: export VAR=value, env, printenv, unset, viewing vs setting, persistence across sessions",
            "shell config files: .bashrc vs .bash_profile vs .profile, source/dot command, /etc/profile.d/, login vs non-login shells",
            "PATH management: prepend/append, persistent PATH in profile files, which, type, hash -r, update-alternatives",
            "aliases and shell functions: alias/unalias, complex aliases, shell functions in .bashrc, type to check definitions",
            "timezone and locale: timedatectl set-timezone, list-timezones, set-ntp, localectl set-locale, list-locales, LANG, LC_ALL",
            "hostname configuration: hostnamectl set-hostname, /etc/hostname, /etc/hosts, static vs transient vs pretty hostname",
            "system-wide defaults: /etc/environment, /etc/default/, /etc/sysconfig/, update-alternatives --config, systemd-tmpfiles, XDG dirs",
        ],
    },
    # --- 2 ADDITIONAL DOMAINS ---
    "web_servers_services": {
        "count": 800,
        "commands": (
            "nginx (nginx -t, nginx -s reload, sites-available, sites-enabled, proxy_pass, "
            "ssl_certificate, location blocks, upstream, access_log, error_log), "
            "apache2/httpd (a2ensite, a2dissite, a2enmod, a2dismod, apachectl, "
            "VirtualHost, ProxyPass, .htaccess, mod_rewrite, mod_ssl), "
            "systemd service files (ExecStart, Restart, WantedBy, After, Environment, "
            "Type=simple/forking/oneshot, systemctl daemon-reload), "
            "certbot (certonly --standalone/--webroot/--nginx/--apache, renew --dry-run), "
            "php-fpm, gunicorn, uwsgi, pm2, supervisord, "
            "curl (-I, -X, -H, -d, -k, -o), ab, wrk, siege (benchmarking), "
            "haproxy, traefik (config basics)"
        ),
        "description": "Nginx, Apache, reverse proxy, SSL/TLS, systemd services, PHP/Python app servers, load balancing",
        "subtopics": [
            "nginx: config syntax, server blocks, location blocks, nginx -t (test), nginx -s reload, sites-available/sites-enabled, access/error logs",
            "nginx reverse proxy: proxy_pass, proxy_set_header, upstream block, load balancing (round-robin, least_conn, ip_hash), websocket proxy",
            "apache: VirtualHost config, a2ensite/a2dissite, a2enmod/a2dismod (rewrite, ssl, proxy), apachectl graceful, .htaccess, mod_rewrite rules",
            "SSL/TLS setup: certbot --nginx/--apache, certbot certonly --standalone/--webroot, renew --dry-run, nginx ssl_certificate directives, redirect HTTP→HTTPS",
            "systemd service files: [Unit] [Service] [Install], ExecStart, Restart=always, WantedBy=multi-user.target, Environment=, After=network.target, daemon-reload",
            "app servers: php-fpm (pool config, socket vs TCP), gunicorn (--workers, --bind, --timeout), uwsgi, pm2 (start, stop, restart, logs, startup), supervisord",
            "testing and benchmarking: curl -I (headers), curl -X POST -H -d, ab -n -c, wrk -t -c -d, siege, checking response codes and latency",
        ],
    },
    "database_cli": {
        "count": 800,
        "commands": (
            "mysql/mariadb (mysql -u -p, CREATE DATABASE, CREATE USER, GRANT, SHOW DATABASES, "
            "SHOW TABLES, DESCRIBE, SELECT, mysqldump, mysql < dump.sql, mysqlcheck), "
            "postgresql (psql -U -d, createdb, dropdb, createuser, pg_dump, pg_restore, "
            "pg_dumpall, \\l, \\dt, \\d+, \\c, GRANT, ALTER, pg_hba.conf, postgresql.conf), "
            "sqlite3 (.tables, .schema, .import, .dump, .mode, .headers, ATTACH), "
            "redis-cli (SET, GET, DEL, KEYS, HSET, HGET, LPUSH, LRANGE, TTL, EXPIRE, "
            "INFO, CONFIG, SAVE, BGSAVE, MONITOR, SELECT, FLUSHDB, redis-benchmark), "
            "mongo/mongosh (show dbs, use db, db.collection.find, insertOne, updateOne, "
            "deleteOne, createIndex, mongodump, mongorestore, mongoexport, mongoimport)"
        ),
        "description": "MySQL, PostgreSQL, SQLite, Redis, MongoDB CLI operations, backups, user management",
        "subtopics": [
            "MySQL/MariaDB: mysql -u root -p, CREATE DATABASE, CREATE USER, GRANT ALL, SHOW DATABASES/TABLES, DESCRIBE, basic SELECT/INSERT/UPDATE/DELETE",
            "MySQL backups and restore: mysqldump (--all-databases, --single-transaction, specific tables), mysql < dump.sql, mysqlcheck --repair/--optimize",
            "PostgreSQL: psql -U -d, createdb, dropdb, createuser, \\l (list dbs), \\dt (tables), \\d+ (describe), \\c (connect), GRANT, ALTER ROLE",
            "PostgreSQL backups: pg_dump (-Fc, -t, --data-only, --schema-only), pg_restore (-d, -c, --jobs), pg_dumpall, pg_hba.conf auth configuration",
            "SQLite3: sqlite3 db.sqlite, .tables, .schema, .dump, .import csv, .mode (csv, column, json), .headers on, ATTACH DATABASE, .quit",
            "Redis: redis-cli, SET/GET/DEL, KEYS pattern, HSET/HGET (hashes), LPUSH/LRANGE (lists), TTL/EXPIRE, INFO, CONFIG SET, SAVE/BGSAVE, MONITOR, SELECT db",
            "MongoDB: mongosh, show dbs, use dbname, db.coll.find/insertOne/updateOne/deleteOne, createIndex, mongodump/mongorestore, mongoexport/mongoimport",
        ],
    },
}

# ---------------------------------------------------------------------------
# Clarification categories — 8 categories × 550 = 4,400 total
# ---------------------------------------------------------------------------

CLARIFICATION_CATEGORIES = {
    "missing_path": {
        "count": 550,
        "description": "User gives a task but doesn't specify which directory or file",
        "examples": (
            "Delete old logs, Clean up temp files, Back up the config, "
            "Compress the data, Check permissions on the folder, "
            "Move the backups, Archive the reports, Restore the file"
        ),
        "subtopics": [
            "file cleanup without specifying location (logs, temp, cache, old files)",
            "backup or archive tasks without specifying source or destination path",
            "permission or ownership changes without specifying target path or directory",
        ],
    },
    "missing_target": {
        "count": 550,
        "description": "User references 'the service', 'the process', 'the container' without naming it",
        "examples": (
            "Restart the service, Kill the process, Check the container logs, "
            "Stop the database, Enable the daemon, Show the port it's using, "
            "Check if it's running, Reload the config"
        ),
        "subtopics": [
            "service management without naming the service (start, stop, restart, status, enable)",
            "process operations without specifying PID, name, or pattern (kill, signal, trace)",
            "container/database operations without specifying container name, ID, or database instance",
        ],
    },
    "missing_scope": {
        "count": 550,
        "description": "User asks for configuration but scope/parameters are undefined",
        "examples": (
            "Set up firewall rules, Configure the network, Set up SSH access, "
            "Create a cron job, Set up log rotation, Configure monitoring, "
            "Set up alerts, Configure backups, Set resource limits"
        ),
        "subtopics": [
            "firewall/network config without specifying ports, IPs, protocols, or zones",
            "scheduled task setup without specifying timing, command, user, or frequency",
            "access/authentication/monitoring setup without specifying users, methods, or thresholds",
        ],
    },
    "dangerous_ambiguity": {
        "count": 550,
        "description": "User asks something destructive without specifying target precisely",
        "examples": (
            "Format the disk, Delete everything in the directory, Wipe the partition, "
            "Reset all permissions, Remove the old kernel, Kill all processes, "
            "Clear the database, Drop the table, Clean up Docker"
        ),
        "subtopics": [
            "destructive disk/partition operations without specifying device (format, wipe, dd, mkfs)",
            "bulk delete or cleanup operations without specifying scope, exclusions, or dry-run first",
            "database/container destructive ops without specifying which database, table, volume, or image",
        ],
    },
    "multi_distro": {
        "count": 550,
        "description": "Task varies by Linux distribution and user hasn't specified which",
        "examples": (
            "Install nginx, Set up Docker, Add a repository, Install the latest Python, "
            "Configure the firewall, Set up a web server, Install Node.js, "
            "Enable a service, Install a desktop environment"
        ),
        "subtopics": [
            "package installation that differs between apt/dnf/pacman/zypper ecosystems",
            "service management that differs between systemd and init systems",
            "firewall tools that differ between distros (ufw vs firewalld vs iptables vs nftables)",
        ],
    },
    "missing_auth_context": {
        "count": 550,
        "description": "User asks about access/auth but doesn't specify method, user, or scope",
        "examples": (
            "Give them access, Set up authentication, Lock the account, "
            "Grant permissions, Revoke access, Add them to the server, "
            "Set up login, Enable two-factor, Allow remote access"
        ),
        "subtopics": [
            "granting access without specifying user, resource, permission level, or method",
            "authentication setup without specifying mechanism (SSH keys, password, LDAP, PAM, 2FA)",
            "permission changes without specifying level (read/write/execute/sudo) or resource type (file/service/database)",
        ],
    },
    "ambiguous_resource": {
        "count": 550,
        "description": "User refers to a resource (port, disk, interface) without identifying it",
        "examples": (
            "Free up the port, Check the disk, Fix the interface, "
            "Mount the drive, Unmount it, Check what's using it, "
            "Resize the volume, Extend the partition, Check the connection"
        ),
        "subtopics": [
            "port operations without specifying port number or protocol (TCP/UDP)",
            "disk/volume/partition operations without specifying device name (/dev/sdX, /dev/nvmeXnYpZ)",
            "network interface or connection operations without specifying which interface or connection",
        ],
    },
    "missing_environment": {
        "count": 550,
        "description": "Task depends on context (local/remote, prod/dev, bare-metal/VM/container)",
        "examples": (
            "Deploy the app, Restart everything, Update the server, "
            "Scale up, Check the health, Migrate the data, "
            "Roll back the changes, Set up the cluster, Configure HA"
        ),
        "subtopics": [
            "deployment without specifying environment (local/staging/prod), method (docker/systemd/k8s), or host",
            "scaling or clustering without specifying infrastructure type (VM/container/bare-metal) or orchestrator",
            "migration or rollback without specifying source, destination, data format, or what to preserve",
        ],
    },
}

# ---------------------------------------------------------------------------
# Troubleshooting categories — 10 categories × 500 = 5,000
# User reports something broken → one diagnostic/fix command
# ---------------------------------------------------------------------------

TROUBLESHOOTING_CATEGORIES = {
    "service_failures": {
        "count": 500,
        "description": "A service won't start, keeps crashing, or behaves unexpectedly",
        "examples": (
            "nginx won't start, mysql keeps crashing, apache is returning 503, "
            "systemd service fails on boot, docker daemon not starting, "
            "cron jobs aren't running, php-fpm is unresponsive"
        ),
        "subtopics": [
            "service won't start: check journalctl -u service, systemctl status (exit code, main PID), config test commands (nginx -t, apachectl configtest)",
            "service keeps crashing/restarting: journalctl --since '1 hour ago', check OOM killer in dmesg, check resource limits, coredumpctl list",
            "service running but not responding: check listening ports with ss -tlnp, check logs, verify config syntax, check socket/pid file permissions",
            "scheduled tasks not running: crontab -l, check /var/log/syslog for CRON, verify script permissions (+x), check PATH in cron env",
            "slow service: check CPU/memory with top, check I/O with iotop, check connections with ss -s, review access/error logs",
        ],
    },
    "network_connectivity": {
        "count": 500,
        "description": "Can't connect to a host, port, or service over the network",
        "examples": (
            "can't reach the server, connection refused on port 443, "
            "SSH connection timeout, website not loading, "
            "can't ping external hosts, intermittent packet loss"
        ),
        "subtopics": [
            "complete connectivity loss: ping gateway, ping 8.8.8.8, check ip addr (interface up?), check ip route (default route?), check /etc/resolv.conf",
            "connection refused: ss -tlnp | grep :port (anything listening?), check firewall, verify service is running, check bind address (0.0.0.0 vs 127.0.0.1)",
            "connection timeout: traceroute/mtr to destination, check firewall blocking, test alternate port with nc -zv host port",
            "DNS resolution failures: dig domain, cat /etc/resolv.conf, resolvectl status, try dig @8.8.8.8 domain (bypass local DNS)",
            "intermittent issues: mtr --report host (packet loss per hop), ping -c 100 (loss %), check interface errors with ip -s link, dmesg for NIC errors",
        ],
    },
    "disk_space_issues": {
        "count": 500,
        "description": "Disk full, can't write files, quota exceeded, inode exhaustion",
        "examples": (
            "disk is full, no space left on device, can't write to /tmp, "
            "database won't start because disk is full, "
            "inode count exhausted, journal taking too much space"
        ),
        "subtopics": [
            "find what's using space: df -h, du -sh /* | sort -rh | head -20, du --max-depth=2 /var | sort -rn, ncdu /",
            "common space hogs: journalctl --disk-usage + --vacuum-size=500M, docker system df + prune, find /tmp -atime +7, apt clean, old kernels",
            "inode exhaustion: df -i, find / -xdev -type f | cut -d/ -f2-3 | sort | uniq -c | sort -rn (dirs with most files), small file cleanup",
            "emergency space recovery: truncate -s 0 /var/log/large.log, find /var/log -name '*.gz' -delete, du -sh /var/cache/*",
            "prevention: logrotate config, disk usage alerts, LVM extending, monitoring with df in cron",
        ],
    },
    "permission_errors": {
        "count": 500,
        "description": "Permission denied, can't access files, ownership issues, SELinux blocks",
        "examples": (
            "permission denied when running script, can't write to directory, "
            "service can't read config file, user can't access shared folder, "
            "SELinux blocking access, operation not permitted even as root"
        ),
        "subtopics": [
            "basic permission denied: ls -la file (check owner/group/perms), whoami, id, check parent directory execute bit, chmod/chown fix",
            "script execution: check +x bit, file encoding (dos2unix), shebang line, mount options (noexec?), read-only filesystem",
            "service permissions: check service user (systemctl show -p User), config file readable, socket/pid permissions, port < 1024 needs root/capability",
            "SELinux/AppArmor: getenforce, ausearch -m avc -ts recent, audit2why, restorecon -Rv, aa-status, dmesg | grep apparmor",
            "special: chattr +i (immutable), lsattr, mount -o remount,rw, capability issues (getcap/setcap), ACL conflicts",
        ],
    },
    "package_install_failures": {
        "count": 500,
        "description": "Package won't install, dependency conflicts, broken packages, repo errors",
        "examples": (
            "apt install fails with unmet dependencies, dpkg was interrupted, "
            "yum returns GPG check failed, pip install can't find package, "
            "broken packages prevent updates, repository 404 errors"
        ),
        "subtopics": [
            "dependency conflicts: apt --fix-broken install, dpkg --configure -a, aptitude (interactive resolver), dnf --best --allowerasing",
            "broken package state: dpkg --configure -a, apt clean && apt update, dpkg --remove --force-remove-reinstreq, rpm --rebuilddb",
            "repository errors: apt update (check which repos fail), expired keys, add-apt-repository --remove, dnf repolist",
            "pip/npm failures: pip install --no-cache-dir, npm cache clean --force, rm -rf node_modules && npm install",
            "version conflicts: apt list -a package, apt install package=version, pip install 'package>=1.0,<2.0', apt-mark hold",
        ],
    },
    "ssh_problems": {
        "count": 500,
        "description": "SSH connection fails, key rejected, timeout, auth errors, config issues",
        "examples": (
            "SSH connection refused, key is rejected, permission denied publickey, "
            "SSH hangs on connect, too many authentication failures, "
            "can't SSH as root, host key verification failed"
        ),
        "subtopics": [
            "connection refused/timeout: ssh -vvv (verbose debug), check sshd running, check port (ss -tlnp | grep :22), check firewall",
            "key authentication: check permissions (~/.ssh 700, keys 600), ssh-add -l, check authorized_keys format, sshd_config PubkeyAuthentication",
            "password auth: sshd_config PasswordAuthentication, check /var/log/auth.log, check PAM, check user shell (not nologin), AllowUsers/DenyUsers",
            "host key issues: ssh-keygen -R host, StrictHostKeyChecking, 'REMOTE HOST IDENTIFICATION HAS CHANGED' (reinstall or MITM)",
            "'too many auth failures': ssh -o IdentitiesOnly=yes -i key, ssh-add -D (clear agent), MaxAuthTries, fail2ban-client status sshd",
        ],
    },
    "high_resource_usage": {
        "count": 500,
        "description": "High CPU, memory leak, system slow, load spikes, OOM kills",
        "examples": (
            "server is very slow, CPU at 100%, running out of memory, "
            "OOM killer keeps killing processes, load average is very high, "
            "system is swapping heavily, a process is eating all memory"
        ),
        "subtopics": [
            "high CPU: top -bn1 -o %CPU | head -20, ps aux --sort=-%cpu | head -10, mpstat -P ALL 1 5, check iowait, strace -cp PID",
            "high memory: free -h, ps aux --sort=-%mem | head -10, smem -tk, dmesg | grep -i 'oom', slabtop",
            "high load: uptime vs nproc, check %wa iowait, if iowait high: iotop -o, check D-state processes (ps aux | awk '$8 ~ /D/')",
            "heavy swapping: vmstat 1 (si/so columns), swapon --show, sysctl vm.swappiness, find memory hog",
            "OOM killer: dmesg | grep -i 'oom\\|killed process', /proc/PID/oom_score_adj, systemd MemoryMax, increase RAM or reduce workload",
        ],
    },
    "boot_startup_issues": {
        "count": 500,
        "description": "System won't boot, stuck at boot, GRUB errors, kernel panic, fsck failures",
        "examples": (
            "system stuck at boot, GRUB rescue prompt, kernel panic, "
            "emergency mode after update, filesystem check failed, "
            "can't boot after kernel upgrade, initramfs prompt"
        ),
        "subtopics": [
            "GRUB issues: check /boot (df -h /boot full?), grub-install, update-grub, boot from GRUB rescue (set root, linux, initrd), remove old kernels",
            "emergency/rescue mode: journalctl -xb (boot errors), systemctl --failed, check /etc/fstab (wrong UUID?), mount -o remount,rw /",
            "kernel panic: boot previous kernel from GRUB, check dmesg from previous boot (journalctl -b -1), update-initramfs -u, check /boot space",
            "filesystem errors: fsck /dev/sdX (from live USB), e2fsck -f -y, xfs_repair, check SMART (smartctl -a), verify fstab UUIDs with blkid",
            "post-update failures: boot old kernel, dpkg --configure -a, apt --fix-broken install, regenerate initramfs, check module loading in dmesg",
        ],
    },
    "docker_issues": {
        "count": 500,
        "description": "Container won't start, image pull fails, networking issues, volume problems",
        "examples": (
            "docker container exits immediately, can't pull image, "
            "container can't reach the internet, port not accessible, "
            "volume permissions wrong, docker daemon won't start, "
            "compose up fails, container runs out of disk"
        ),
        "subtopics": [
            "container won't start: docker logs container, docker inspect (ExitCode, Error, OOMKilled), docker run without -d, check CMD/ENTRYPOINT, port already in use",
            "image issues: check internet/DNS, docker login (auth), docker system df + prune (disk space), check tag/registry URL, build failures (Dockerfile syntax)",
            "networking: docker network inspect bridge (DNS settings), port mapping not working (docker port, check host firewall), containers can't talk (use custom network)",
            "volume permissions: files created as root, --user flag, USER in Dockerfile, :z/:Z for SELinux, docker volume inspect (Mountpoint)",
            "docker daemon: systemctl status docker, journalctl -u docker, /var/lib/docker disk space, /etc/docker/daemon.json syntax, socket permissions, usermod -aG docker user",
        ],
    },
    "database_issues": {
        "count": 500,
        "description": "Database won't start, connection refused, slow queries, replication lag, corruption",
        "examples": (
            "MySQL won't start after crash, can't connect to PostgreSQL, "
            "queries running very slow, replication is behind, "
            "table is corrupted, connection limit reached, Redis out of memory"
        ),
        "subtopics": [
            "database won't start: check logs (journalctl -u mysql/postgresql, error.log), check disk space (df -h), check data dir permissions, check port conflicts (ss -tlnp)",
            "connection issues: check service running, check listening port, check auth config (pg_hba.conf, mysql.user), check firewall, test local connection first, check max_connections",
            "slow queries: MySQL slow query log, EXPLAIN SELECT, SHOW PROCESSLIST, PostgreSQL pg_stat_statements, check missing indexes, SHOW ENGINE INNODB STATUS",
            "data recovery: mysqlcheck --repair, CHECK TABLE/REPAIR TABLE, pg_resetwal, redis-check-rdb/aof, check backups, binary log point-in-time recovery",
            "resource issues: MySQL innodb_buffer_pool_size, PostgreSQL shared_buffers, connection pooling (pgBouncer), Redis maxmemory eviction policy, SHOW STATUS / pg_stat_activity",
        ],
    },
}

# ---------------------------------------------------------------------------
# Prompt templates — with subtopic support for diversity
# ---------------------------------------------------------------------------

DIRECT_PROMPT = """\
Generate exactly {count} Linux command training examples for the "{domain}" domain.

Domain: {description}
Commands to cover: {commands}
Specific focus for this batch: {subtopic}

Each example is a JSON object with "user" and "assistant" fields.

Format:
- "user": Natural human question. Vary phrasing: "how do I...", "show me...", \
"what command...", "list the...", "I need to...", "can you...", "what's the best way to...", etc.
- "assistant": A bash code block with ONE command, followed by exactly one brief \
explanation sentence.

Example:
{{"user": "How do I find all PDF files in my home directory?", \
"assistant": "```bash\\nfind ~/ -name '*.pdf' -type f\\n```\\n\
Searches your home directory recursively for all files ending in .pdf."}}

Rules:
1. ONE command per example — never list alternatives or variations
2. Use realistic values: real paths (/home/admin, /var/log/nginx, /etc/ssh), \
real usernames (admin, deploy, jsmith, webapp), real IPs (192.168.1.50, 10.0.0.1)
3. Never use {{{{placeholder}}}} or template syntax like <file> or [path]
4. The explanation must be exactly ONE sentence — do not restate the question
5. Cover diverse use cases — every example should be meaningfully different
6. Include both simple and intermediate-complexity commands
7. Commands must be correct and production-ready
8. Focus specifically on the sub-topic for this batch

Return ONLY a valid JSON array of {count} objects. No markdown fences, no extra text."""

CLARIFICATION_PROMPT = """\
Generate exactly {count} examples where a user asks an ambiguous Linux task \
and the assistant asks ONE clarifying question instead of guessing.

Category: {category} — {description}
Example requests like: {examples}
Specific focus for this batch: {subtopic}

Each example is a JSON object with "user" and "assistant" fields.

Format:
- "user": A natural but ambiguous request missing critical information
- "assistant": ONE clarifying question, optionally suggesting a diagnostic command

Examples:
{{"user": "Delete the old logs", \
"assistant": "Which directory should I look in? For system logs check /var/log/, \
or specify your application's log path."}}
{{"user": "Restart the service", \
"assistant": "Which service? Run `systemctl list-units --type=service --state=running` \
to see active services."}}
{{"user": "Free up the port", \
"assistant": "Which port number? Run `ss -tlnp` to see which ports are currently in use \
and what's listening on them."}}

Rules:
1. The request must be genuinely ambiguous — missing a critical detail
2. Ask exactly ONE focused clarifying question
3. Optionally suggest a diagnostic command to help the user answer
4. Never guess or provide a command based on assumptions
5. Vary the phrasing — don't start every response the same way
6. Cover diverse scenarios — every example should be meaningfully different
7. Focus specifically on the sub-topic for this batch

Return ONLY a valid JSON array of {count} objects. No markdown fences, no extra text."""

TROUBLESHOOT_PROMPT = """\
Generate exactly {count} examples where a user reports a Linux problem and \
the assistant gives ONE diagnostic or fix command.

Category: {category} — {description}
Example problem reports: {examples}
Specific focus for this batch: {subtopic}

Each example is a JSON object with "user" and "assistant" fields.

The user reports a PROBLEM (something broken, slow, or failing). \
The assistant gives ONE command to diagnose or fix it, plus a brief explanation.

Format:
- "user": Natural problem description like: "X won't start", "can't connect to Y", \
"getting error Z", "system is slow", "disk is full"
- "assistant": A ```bash code block with ONE diagnostic/fix command, followed by one explanation sentence.

Examples:
{{"user": "nginx won't start after I edited the config", \
"assistant": "```bash\\nnginx -t\\n```\\nTests the nginx configuration for syntax errors and reports the exact line causing the failure."}}
{{"user": "my disk is completely full and I can't write anything", \
"assistant": "```bash\\ndu -sh /* 2>/dev/null | sort -rh | head -10\\n```\\nShows the 10 largest top-level directories to identify what's consuming the most space."}}
{{"user": "I keep getting connection refused on port 8080", \
"assistant": "```bash\\nss -tlnp | grep :8080\\n```\\nChecks if anything is actually listening on port 8080."}}

Rules:
1. ONE command per example — prefer a diagnostic command that reveals the root cause
2. Use realistic values: real paths, service names, port numbers, log locations
3. Never use {{{{placeholder}}}} or template syntax like <file> or [path]
4. The explanation should say what the command reveals or fixes, in ONE sentence
5. Every example must be meaningfully different
6. Prefer diagnostic commands first (check before blindly fixing)
7. Include sudo when the command requires root privileges
8. Focus specifically on the sub-topic for this batch

Return ONLY a valid JSON array of {count} objects. No markdown fences, no extra text."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_chatml(user_msg: str, assistant_msg: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": NEW_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg.strip()},
            {"role": "assistant", "content": assistant_msg.strip()},
        ]
    }


def parse_json_response(text: str) -> list[dict] | None:
    """Extract JSON array from API response, handling markdown fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    try:
        result = json.loads(text.strip())
        if isinstance(result, list):
            return result
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def save_jsonl(data: list[dict], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} examples to {filepath}")


def append_jsonl(data: list[dict], filepath: Path):
    """Append examples to JSONL file (for incremental saving)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def call_api(client, prompt: str, label: str, max_retries: int = 2) -> list[dict]:
    """Call Claude API and parse the JSON response."""
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** (attempt + 1)
                print(f"\n    API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"\n    ERROR: API call failed for {label}: {e}")
            return []

        parsed = parse_json_response(text)
        if parsed:
            return parsed

        if attempt < max_retries:
            print(f"\n    Retry {attempt + 1}/{max_retries} for {label} (failed to parse JSON)")
            time.sleep(1)

    print(f"\n    WARNING: Failed to parse response for {label} after {max_retries + 1} attempts")
    return []


def get_subtopic(subtopics: list[str], batch_idx: int) -> str:
    """Cycle through subtopics for diversity across batches."""
    return subtopics[batch_idx % len(subtopics)]


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def count_existing_lines(filepath: Path) -> int:
    """Count lines in existing JSONL file for resume support."""
    if not filepath.exists():
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def generate_direct_examples(client, dry_run: bool = False, resume: bool = False) -> list[dict]:
    """Generate direct-answer training examples across all domains."""
    print("\n[1/3] Generating direct-answer examples...")
    all_examples = []
    output_path = RAW_DIR / "synthetic_direct.jsonl"

    existing_count = 0
    if resume and not dry_run:
        existing_count = count_existing_lines(output_path)
        if existing_count > 0:
            print(f"  RESUME: Found {existing_count} existing examples, skipping completed batches...")
    elif not dry_run and output_path.exists():
        output_path.unlink()

    # Calculate how many batches to skip based on existing line count
    skip_remaining = existing_count

    total_domains = len(DIRECT_DOMAINS)
    for domain_idx, (domain_name, info) in enumerate(DIRECT_DOMAINS.items(), 1):
        total = info["count"]
        batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        subtopics = info.get("subtopics", [info["description"]])

        print(f"\n  [{domain_idx}/{total_domains}] {domain_name} ({total} examples, {batches} batches)")

        for batch_idx in range(batches):
            count = min(BATCH_SIZE, total - batch_idx * BATCH_SIZE)

            # Skip batches covered by existing data
            if skip_remaining >= count:
                skip_remaining -= count
                print(f"    Batch {batch_idx + 1}/{batches} — skipped (already generated)")
                continue
            skip_remaining = 0

            subtopic = get_subtopic(subtopics, batch_idx)
            prompt = DIRECT_PROMPT.format(
                count=count,
                domain=domain_name.replace("_", " "),
                description=info["description"],
                commands=info["commands"],
                subtopic=subtopic,
            )

            label = f"{domain_name} batch {batch_idx + 1}/{batches}"

            if dry_run:
                print(f"    [DRY RUN] {label}: {count} examples — focus: {subtopic[:60]}")
                continue

            print(f"    Batch {batch_idx + 1}/{batches} ({count} ex, focus: {subtopic[:50]})...", end=" ", flush=True)
            examples = call_api(client, prompt, label)

            chatml_examples = []
            for ex in examples:
                user_msg = ex.get("user", "")
                asst_msg = ex.get("assistant", "")
                if user_msg and asst_msg:
                    chatml_examples.append(to_chatml(user_msg, asst_msg))

            all_examples.extend(chatml_examples)
            if chatml_examples:
                append_jsonl(chatml_examples, output_path)
            print(f"got {len(chatml_examples)}")

            time.sleep(0.3)

    print(f"\n  Total direct-answer examples: {len(all_examples)}")
    return all_examples


def generate_clarification_examples(client, dry_run: bool = False, resume: bool = False) -> list[dict]:
    """Generate clarification dialogue training examples."""
    print("\n[2/3] Generating clarification dialogues...")
    all_examples = []
    output_path = RAW_DIR / "synthetic_clarification.jsonl"

    existing_count = 0
    if resume and not dry_run:
        existing_count = count_existing_lines(output_path)
        if existing_count > 0:
            print(f"  RESUME: Found {existing_count} existing examples, skipping completed batches...")
    elif not dry_run and output_path.exists():
        output_path.unlink()

    skip_remaining = existing_count

    total_categories = len(CLARIFICATION_CATEGORIES)
    for cat_idx, (cat_name, info) in enumerate(CLARIFICATION_CATEGORIES.items(), 1):
        total = info["count"]
        batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        subtopics = info.get("subtopics", [info["description"]])

        print(f"\n  [{cat_idx}/{total_categories}] {cat_name} ({total} examples, {batches} batches)")

        for batch_idx in range(batches):
            count = min(BATCH_SIZE, total - batch_idx * BATCH_SIZE)

            if skip_remaining >= count:
                skip_remaining -= count
                print(f"    Batch {batch_idx + 1}/{batches} — skipped (already generated)")
                continue
            skip_remaining = 0

            subtopic = get_subtopic(subtopics, batch_idx)
            prompt = CLARIFICATION_PROMPT.format(
                count=count,
                category=cat_name.replace("_", " "),
                description=info["description"],
                examples=info["examples"],
                subtopic=subtopic,
            )

            label = f"{cat_name} batch {batch_idx + 1}/{batches}"

            if dry_run:
                print(f"    [DRY RUN] {label}: {count} examples — focus: {subtopic[:60]}")
                continue

            print(f"    Batch {batch_idx + 1}/{batches} ({count} ex, focus: {subtopic[:50]})...", end=" ", flush=True)
            examples = call_api(client, prompt, label)

            chatml_examples = []
            for ex in examples:
                user_msg = ex.get("user", "")
                asst_msg = ex.get("assistant", "")
                if user_msg and asst_msg:
                    chatml_examples.append(to_chatml(user_msg, asst_msg))

            all_examples.extend(chatml_examples)
            if chatml_examples:
                append_jsonl(chatml_examples, output_path)
            print(f"got {len(chatml_examples)}")

            time.sleep(0.3)

    print(f"\n  Total clarification examples: {len(all_examples)}")
    return all_examples


def generate_troubleshoot_examples(client, dry_run: bool = False, resume: bool = False) -> list[dict]:
    """Generate troubleshooting training examples."""
    print("\n[3/3] Generating troubleshooting examples...")
    all_examples = []
    output_path = RAW_DIR / "synthetic_troubleshoot.jsonl"

    existing_count = 0
    if resume and not dry_run:
        existing_count = count_existing_lines(output_path)
        if existing_count > 0:
            print(f"  RESUME: Found {existing_count} existing examples, skipping completed batches...")
    elif not dry_run and output_path.exists():
        output_path.unlink()

    skip_remaining = existing_count

    total_categories = len(TROUBLESHOOTING_CATEGORIES)
    for cat_idx, (cat_name, info) in enumerate(TROUBLESHOOTING_CATEGORIES.items(), 1):
        total = info["count"]
        batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        subtopics = info.get("subtopics", [info["description"]])

        print(f"\n  [{cat_idx}/{total_categories}] {cat_name} ({total} examples, {batches} batches)")

        for batch_idx in range(batches):
            count = min(BATCH_SIZE, total - batch_idx * BATCH_SIZE)

            if skip_remaining >= count:
                skip_remaining -= count
                print(f"    Batch {batch_idx + 1}/{batches} — skipped (already generated)")
                continue
            skip_remaining = 0

            subtopic = get_subtopic(subtopics, batch_idx)
            prompt = TROUBLESHOOT_PROMPT.format(
                count=count,
                category=cat_name.replace("_", " "),
                description=info["description"],
                examples=info["examples"],
                subtopic=subtopic,
            )

            label = f"{cat_name} batch {batch_idx + 1}/{batches}"

            if dry_run:
                print(f"    [DRY RUN] {label}: {count} examples — focus: {subtopic[:60]}")
                continue

            print(f"    Batch {batch_idx + 1}/{batches} ({count} ex, focus: {subtopic[:50]})...", end=" ", flush=True)
            examples = call_api(client, prompt, label)

            chatml_examples = []
            for ex in examples:
                user_msg = ex.get("user", "")
                asst_msg = ex.get("assistant", "")
                if user_msg and asst_msg:
                    chatml_examples.append(to_chatml(user_msg, asst_msg))

            all_examples.extend(chatml_examples)
            if chatml_examples:
                append_jsonl(chatml_examples, output_path)
            print(f"got {len(chatml_examples)}")

            time.sleep(0.3)

    print(f"\n  Total troubleshooting examples: {len(all_examples)}")
    return all_examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data via Claude API"
    )
    parser.add_argument("--direct-only", action="store_true",
                        help="Only generate direct-answer examples")
    parser.add_argument("--clarification-only", action="store_true",
                        help="Only generate clarification examples")
    parser.add_argument("--troubleshoot-only", action="store_true",
                        help="Only generate troubleshooting examples")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without API calls")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from where a previous run left off (skip existing data)")
    args = parser.parse_args()

    # Determine which categories to generate
    generate_all = not (args.direct_only or args.clarification_only or args.troubleshoot_only)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    client = None
    if not args.dry_run:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)

    direct_examples = []
    clarification_examples = []
    troubleshoot_examples = []

    if generate_all or args.direct_only:
        direct_examples = generate_direct_examples(client, args.dry_run, args.resume)

    if generate_all or args.clarification_only:
        clarification_examples = generate_clarification_examples(client, args.dry_run, args.resume)

    if generate_all or args.troubleshoot_only:
        troubleshoot_examples = generate_troubleshoot_examples(client, args.dry_run, args.resume)

    if args.dry_run:
        total_direct = sum(d["count"] for d in DIRECT_DOMAINS.values()) if (generate_all or args.direct_only) else 0
        total_clarification = sum(c["count"] for c in CLARIFICATION_CATEGORIES.values()) if (generate_all or args.clarification_only) else 0
        total_troubleshoot = sum(t["count"] for t in TROUBLESHOOTING_CATEGORIES.values()) if (generate_all or args.troubleshoot_only) else 0
        total_batches = 0
        if generate_all or args.direct_only:
            total_batches += sum((d["count"] + BATCH_SIZE - 1) // BATCH_SIZE for d in DIRECT_DOMAINS.values())
        if generate_all or args.clarification_only:
            total_batches += sum((c["count"] + BATCH_SIZE - 1) // BATCH_SIZE for c in CLARIFICATION_CATEGORIES.values())
        if generate_all or args.troubleshoot_only:
            total_batches += sum((t["count"] + BATCH_SIZE - 1) // BATCH_SIZE for t in TROUBLESHOOTING_CATEGORIES.values())
        total_examples = total_direct + total_clarification + total_troubleshoot
        print(f"\n{'='*60}")
        print(f"DRY RUN SUMMARY")
        print(f"  Direct-answer:    {total_direct:>6}")
        print(f"  Clarification:    {total_clarification:>6}")
        print(f"  Troubleshooting:  {total_troubleshoot:>6}")
        print(f"  ─────────────────────────")
        print(f"  Total target:     {total_examples:>6}")
        print(f"  Total API calls:  {total_batches:>6}")
        print(f"  Estimated cost:   ~$7.00-8.00")
        print(f"{'='*60}")
        return

    total = len(direct_examples) + len(clarification_examples) + len(troubleshoot_examples)
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"  Direct-answer:    {len(direct_examples)} examples")
    print(f"  Clarification:    {len(clarification_examples)} examples")
    print(f"  Troubleshooting:  {len(troubleshoot_examples)} examples")
    print(f"  Total: {total} examples")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  python reformat_data.py          # combine with existing data")
    print(f"  python filter_data.py --input raw/combined_directstyle.jsonl --output ../output/ --split")


if __name__ == "__main__":
    main()
