#!/usr/bin/env python3
import subprocess
import sys


def main() -> None:
    out_path, lib_path = sys.argv[1], sys.argv[2]
    dump = subprocess.run(
        ["dumpbin", "/NOLOGO", "/SYMBOLS", lib_path],
        capture_output=True, text=True, check=True,
    ).stdout

    # gather all objects and mark them for export
    functions: set[str] = set()
    data: set[str] = set()
    for line in dump.splitlines():
        if "|" not in line:
            continue
        head, name = line.split("|", 1)
        fields = head.split()
        if "External" not in fields or fields[2] == "UNDEF":
            continue
        name = name.split()[0]
        # skip imports and deleting destructors (??_G/??_E), which trip LNK4102
        if (name.startswith(("__imp_", "??_G", "??_E"))
                or name in ("@feat.00", "@comp.id")):
            continue
        (functions if "()" in fields else data).add(name)

    data -= functions

    with open(out_path, "w") as out:
        out.write("EXPORTS\n")
        for name in sorted(functions):
            out.write(f"    {name}\n")
        for name in sorted(data):
            out.write(f"    {name} DATA\n")


if __name__ == "__main__":
    main()
