#!/bin/env python3

"""
Compiles a GLSL shader "filename.ext" and
outputs a c compatible file that contains

- an array uint32_t filename_ext_spv[].
- a function uint32_t *merian_filename_ext_spv(void)
- a function uint32_t merian_filename_ext_spv_size(void)

Usage: python compile_shader.py in.ext [output.h]
"""

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path


def to_int_array(b_array: bytes):
    # This must be true for SPIR-V by spec.
    assert len(b_array) % 4 == 0
    result = []
    for i in range(0, len(b_array), 4):
        result.append(int.from_bytes(b_array[i : i + 4], byteorder="little"))
    return result


def compile_shader(input_path, glslc_args, optimize) -> bytes:
    with tempfile.TemporaryDirectory() as tempdir:
        shader = Path(tempdir) / "shader.spv"
        glslc_command = (
            ["glslangValidator", "-V"] + glslc_args + ["-o", shader, input_path]
        )
        subprocess.check_call(glslc_command)

        if optimize:
            spirv_opt_command = [
                "spirv-opt",
                "--scalar-block-layout",
                "-O",
                "-o",
                shader,
                shader,
            ]
            subprocess.check_call(spirv_opt_command)

        with open(shader, "rb") as f:
            return f.read()


def main():
    parser = argparse.ArgumentParser("compile_shader.py")
    parser.add_argument("shader")
    parser.add_argument("out")
    parser.add_argument("glslc_args", nargs=argparse.REMAINDER)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--name")
    args = parser.parse_args()

    in_path = Path(args.shader)
    shader_name = args.name if args.name else re.sub(r"[^A-Za-z0-9]", "_", in_path.name)

    header = """\
#pragma once

#include "stdint.h"

#if __cplusplus
extern "C" {{
#endif

const uint32_t* merian_{shader_name}_spv(void);

uint32_t merian_{shader_name}_spv_size(void);

#if __cplusplus
}}
#endif
""".format(shader_name=shader_name)

    header_path = os.path.join(args.out, in_path.name + ".spv.h")
    with open(header_path, "w") as f:
        f.write(header)
        # print(f"wrote header to {header_path}")

    spv = compile_shader(in_path, args.glslc_args, args.optimize)
    implementation = """\
#include "stdint.h"

static const uint32_t {shader_name}_spv[] = {{
    {spv_int_array}
}};

const uint32_t* merian_{shader_name}_spv(void) {{
    return {shader_name}_spv;
}}

uint32_t merian_{shader_name}_spv_size(void) {{
    return sizeof({shader_name}_spv);
}}

""".format(
        shader_name=shader_name,
        spv_int_array=", ".join(f"{s:#010x}" for s in to_int_array(spv)),
    )

    impl_path = Path(os.path.join(args.out, in_path.name + ".spv.c"))
    with open(impl_path, "w") as f:
        f.write(implementation)
        # print(f"wrote implementation to {impl_path}")

    if "--depfile" in args.glslc_args:
        depfile_path = args.glslc_args[args.glslc_args.index("--depfile") + 1]
        # print(f"fixup depfile {depfile_path}")
        with open(depfile_path, "r") as f:
            depfile = f.read()
        depfile = re.sub("^(.*):", impl_path.name + ":", depfile)
        with open(depfile_path, "w") as f:
            f.write(depfile)


if __name__ == "__main__":
    main()
