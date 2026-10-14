#!/usr/bin/env python3
"""Read-only Android hardware probe. This does not run AEIC decoding."""

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path


PROPERTIES = {
    "model": "ro.product.model",
    "android_version": "ro.build.version.release",
    "api": "ro.build.version.sdk",
    "abi_list": "ro.product.cpu.abilist",
    "primary_abi": "ro.product.cpu.abi",
    "secondary_abi": "ro.product.cpu.abi2",
    "heap_growth_limit": "dalvik.vm.heapgrowthlimit",
    "heap_size": "dalvik.vm.heapsize",
    "low_ram": "ro.config.low_ram",
}
LIBRARIES = [
    directory + filename
    for directory in (
        "/system/lib/", "/system/lib64/", "/vendor/lib/",
        "/vendor/lib64/", "/system/vendor/lib/", "/system/vendor/lib64/",
    )
    for filename in ("libOpenCL.so", "libGLESv2.so")
] + [
    directory + filename
    for directory in (
        "/system/lib/egl/", "/system/lib64/egl/", "/vendor/lib/egl/",
        "/vendor/lib64/egl/", "/system/vendor/lib/egl/",
        "/system/vendor/lib64/egl/",
    )
    for filename in ("libGLES_mali.so", "libGLESv2_adreno.so")
]


def positive_timeout(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number")
    return seconds


class Adb:
    def __init__(self, executable, timeout):
        self.executable = executable
        self.timeout = timeout
        self.serial = None
        self.errors = []

    def run(self, arguments, optional=False):
        command = [self.executable]
        if self.serial is not None:
            command += ["-s", self.serial]
        command += arguments
        started = time.monotonic()
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=self.timeout, check=False,
            )
            record = {
                "command": arguments, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip(),
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            record = {
                "command": arguments, "returncode": None,
                "stdout": "", "stderr": str(error),
            }
        record["seconds"] = round(time.monotonic() - started, 3)
        if record["returncode"] != 0 and not optional:
            self.errors.append(record)
        return record

    def shell(self, *arguments, optional=False):
        return self.run(["shell", *arguments], optional=optional)


def select_device(adb, requested_serial):
    discovery = adb.run(["devices"])
    if discovery["returncode"] != 0:
        raise RuntimeError("ADB discovery failed: " + discovery["stderr"])
    devices = []
    for line in discovery["stdout"].splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1] in (
            "device", "unauthorized", "offline", "recovery", "sideload",
        ):
            devices.append({"serial": fields[0], "state": fields[1]})
    authorized = [item for item in devices if item["state"] == "device"]
    if requested_serial is not None:
        if not any(item["serial"] == requested_serial for item in authorized):
            raise RuntimeError("Requested serial is not a connected, authorized device.")
        adb.serial = requested_serial
    elif len(authorized) == 1:
        adb.serial = authorized[0]["serial"]
    elif not authorized:
        raise RuntimeError("No connected, authorized Android device; check USB authorization.")
    else:
        raise RuntimeError("Multiple authorized devices; specify --serial explicitly: " +
                           ", ".join(item["serial"] for item in authorized))
    return devices


def parse_memory(text):
    fields = {
        name: int(value) * 1024
        for name, value in re.findall(r"^(\w+):\s+(\d+)\s+kB", text, re.M)
    }
    available = fields.get("MemAvailable")
    method = "MemAvailable"
    if available is None:
        names = ("MemFree", "Buffers", "Cached")
        available = sum(fields[name] for name in names) if all(
            name in fields for name in names
        ) else None
        method = "MemFree + Buffers + Cached (rough fallback)"
    return {"total_bytes": fields.get("MemTotal"),
            "available_bytes": available, "available_method": method}


def parse_limits(text):
    limits = {}
    for label in ("Max resident set", "Max address space"):
        match = re.search(r"^" + re.escape(label) + r"\s+(\S+)\s+(\S+)\s*(.*)$", text, re.M)
        if match:
            limits[label] = {"soft": match[1], "hard": match[2], "units": match[3]}
    return limits


def collect(adb):
    properties = {}
    for name, prop in PROPERTIES.items():
        result = adb.shell("getprop", prop)
        properties[name] = result["stdout"] if result["returncode"] == 0 else None
    abi_text = properties.get("abi_list") or ""
    abis = [abi.strip() for abi in abi_text.split(",") if abi.strip()]
    if not abis:
        abis = list(dict.fromkeys(filter(None, (
            properties.get("primary_abi"), properties.get("secondary_abi"),
        ))))

    cpu = adb.shell("cat", "/proc/cpuinfo")
    feature_lines = re.findall(r"^(?:Features|flags)\s*:\s*(.*)$", cpu["stdout"], re.M | re.I)
    features = sorted(set(" ".join(feature_lines).lower().split()))
    neon = bool(set(features).intersection(("neon", "asimd"))) if features else None
    # Do not retain raw cpuinfo, which can include a hardware serial number.
    memory = adb.shell("cat", "/proc/meminfo")
    limits = adb.shell("cat", "/proc/self/limits")
    storage = adb.shell("df", "/data", "/sdcard")
    libraries = adb.shell("ls", "-l", *LIBRARIES, optional=True)
    present = []
    for line in libraries["stdout"].splitlines():
        if re.search(r"no such|cannot|denied|not found|error", line, re.I):
            continue
        for path in LIBRARIES:
            if re.search(r"(?<!\S)" + re.escape(path) + r"(?=\s|$)", line):
                present.append(path)
    gpu_probe = {
        "paths_present": sorted(set(present)),
        "returncode": libraries["returncode"],
        "diagnostic": libraries["stderr"],
        "note": "Missing targeted libraries are expected. Presence does not prove usable OpenCL.",
    }
    if libraries["returncode"] is None:
        adb.errors.append(libraries)
    return {
        "properties": properties, "abis": abis,
        "cpu": {"features": features, "neon_or_asimd": neon},
        "memory": parse_memory(memory["stdout"]),
        "shell_process_limits": parse_limits(limits["stdout"]),
        "shell_process_limits_raw": limits["stdout"],
        "storage": {"returncode": storage["returncode"], "df": storage["stdout"]},
        "gpu_libraries": gpu_probe,
    }


def mib(value):
    return "unknown" if value is None else "%.0f MiB" % (value / (1024 * 1024))


def print_report(report):
    if "hardware" not in report:
        print("Probe failed: " + report["message"], file=sys.stderr)
        return
    hardware = report["hardware"]
    props = hardware["properties"]
    print("Device: %s (serial %s)" % (props.get("model") or "unknown", report["serial"]))
    print("Android: %s; API %s; ABIs: %s" % (
        props.get("android_version") or "unknown", props.get("api") or "unknown",
        ", ".join(hardware["abis"]) or "unknown",
    ))
    neon = hardware["cpu"]["neon_or_asimd"]
    print("NEON/ASIMD: " + ("unknown" if neon is None else "yes" if neon else "not advertised"))
    memory = hardware["memory"]
    print("RAM: %s total; %s available (%s)" % (
        mib(memory["total_bytes"]), mib(memory["available_bytes"]), memory["available_method"],
    ))
    print("Java heap hints: growth=%s; max=%s; low_ram=%s" % (
        props.get("heap_growth_limit") or "unset", props.get("heap_size") or "unset",
        props.get("low_ram") or "unset",
    ))
    print("Heap hints and shell limits do not establish a native app's memory budget.")
    print("Shell limits: " + (json.dumps(hardware["shell_process_limits"])
          if hardware["shell_process_limits"] else "unknown"))
    print("Storage (df /data /sdcard):\n" + (hardware["storage"]["df"] or "unknown"))
    found = hardware["gpu_libraries"]["paths_present"]
    print("Targeted GPU/OpenCL libraries: " + (", ".join(found) or "none observed"))
    print("Library presence does not prove GPU inference support.")
    api = props.get("api") or ""
    if api.isdigit() and int(api) < 24:
        print("Constraint: no standard Android Vulkan or NNAPI path at this API; test native CPU first.")
    elif api.isdigit() and int(api) < 27:
        print("Constraint: NNAPI is unavailable at this API.")
    if hardware["abis"] and not any("64" in abi for abi in hardware["abis"]):
        print("Constraint: advertised ABIs are 32-bit; large models face address-space limits.")
    print("Read-only hardware probe only; AEIC decoding has not run.")
    if report["errors"]:
        print("Incomplete probe: %d command(s) failed." % len(report["errors"]), file=sys.stderr)
        for error in report["errors"]:
            print("  %s: %s" % (" ".join(error["command"]),
                  error["stderr"] or "exit status %s" % error["returncode"]), file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", default="adb", help="ADB executable path (default: adb)")
    parser.add_argument("--serial", help="Authorized device serial; required if multiple devices are authorized")
    parser.add_argument("--timeout", type=positive_timeout, default=15.0, help="Timeout per ADB command in seconds")
    parser.add_argument("--output", type=Path, help="Optional local JSON output file")
    args = parser.parse_args(argv)
    adb = Adb(args.adb, args.timeout)
    report = {"status": "failed", "errors": adb.errors, "decode_run": False}
    try:
        report["devices"] = select_device(adb, args.serial)
        report["serial"] = adb.serial
        report["hardware"] = collect(adb)
        report["status"] = "partial" if adb.errors else "complete"
    except RuntimeError as error:
        report["message"] = str(error)
    print_report(report)
    if args.output is not None:
        try:
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except OSError as error:
            print("Could not save JSON: " + str(error), file=sys.stderr)
            return 1
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
