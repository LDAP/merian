#!/bin/env python3

"""
Compiles a GLSL "filename.ext" or Slang shader "filename.slang" and
outputs a file "filename.ext.spv.c" that contains

- an array uint32_t filename_ext_spv[].
- a function uint32_t *<prefix>_<name or filename_ext>_spv(void)
- a function uint32_t <prefix>_<name or filename_ext>_spv_size(void)

as well as a "filename.ext.spv.h" containing declarations for the above.

Usage: python compile_shader.py in.ext header.h impl.c
"""

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List


def to_int_array(b_array: bytes):
    # This must be true for SPIR-V by spec.
    assert len(b_array) % 4 == 0
    result = []
    for i in range(0, len(b_array), 4):
        result.append(int.from_bytes(b_array[i : i + 4], byteorder="little"))
    return result


def compile_glsl_shader(args, user_compiler_args: List[str]) -> bytes:
    compiler_args = user_compiler_args.copy()
    if args.depfile:
        compiler_args += ["--depfile", args.depfile]
    compiler_args += ["--target-env", "vulkan1.3", "-V"]

    with tempfile.TemporaryDirectory() as tempdir:
        shader = Path(tempdir) / "shader.spv"
        glslc_command = (
            [args.glslc_path] + compiler_args + ["-o", shader, args.shader_path]
        )
        subprocess.check_call(glslc_command)

        with open(shader, "rb") as f:
            return f.read()


def compile_slang_shader(args, user_compiler_args) -> bytes:
    compiler_args = user_compiler_args.copy()
    if args.depfile:
        compiler_args += ["-depfile", args.depfile]

    with tempfile.TemporaryDirectory() as tempdir:
        shader = Path(tempdir) / "shader.spv"
        glslc_command = (
            [args.slangc_path]
            + compiler_args
            + [args.shader_path, "-target", "spirv", "-o", shader]
        )
        subprocess.check_call(glslc_command)

        with open(shader, "rb") as f:
            return f.read()


def compile_shader(args, compiler_args) -> bytes:
    if args.shader_path.suffix == ".slang":
        return compile_slang_shader(args, compiler_args)
    else:
        return compile_glsl_shader(args, compiler_args)


def main():
    parser = argparse.ArgumentParser("compile_shader.py")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--prefix", default="merian")
    parser.add_argument("--glslc_path", default="glslangValidator")
    parser.add_argument("--slangc_path", default="slangc")
    parser.add_argument("--depfile", type=Path, required=False)
    parser.add_argument("shader_path", type=Path)
    parser.add_argument("header_path", type=Path)
    parser.add_argument("implementation_path", type=Path)
    args, compiler_args = parser.parse_known_args()
    if args.debug:
        print(args)
        print(f"compiler args: {compiler_args}")
        print(f"cwd {os.getcwd()}")

    shader_name = re.sub(
        r"[^A-Za-z0-9]", "_", args.name if args.name else args.shader_path.name
    )
    prefix = re.sub(r"[^A-Za-z0-9]", "_", args.prefix)

    header = """\
#pragma once

#include "stdint.h"

#if __cplusplus
extern "C" {{
#endif

const uint32_t* {prefix}_{shader_name}_spv(void);

uint32_t {prefix}_{shader_name}_spv_size(void);

#if __cplusplus
}}
#endif
""".format(prefix=prefix, shader_name=shader_name)

    with open(args.header_path, "w") as f:
        f.write(header)
        if args.debug:
            print(f"wrote header to {args.header_path}")

    spv = compile_shader(args, compiler_args)

    implementation = """\
#include "stdint.h"

static const uint32_t {shader_name}_spv[] = {{
    {spv_int_array}
}};

const uint32_t* {prefix}_{shader_name}_spv(void) {{
    return {shader_name}_spv;
}}

uint32_t {prefix}_{shader_name}_spv_size(void) {{
    return sizeof({shader_name}_spv);
}}

""".format(
        prefix=prefix,
        shader_name=shader_name,
        spv_int_array=", ".join(f"{s:#010x}" for s in to_int_array(spv)),
    )

    with open(args.implementation_path, "w") as f:
        f.write(implementation)
        if args.debug:
            print(f"wrote implementation to {args.implementation_path}")

    if args.depfile:
        if args.debug:
            print(f"fixup depfile {args.depfile}")
        with open(args.depfile, "r") as f:
            depfile = f.read()
        depfile = re.sub(
            "^(.*):(?![\\\\/])", args.implementation_path.name + ":", depfile
        )
        with open(args.depfile, "w") as f:
            f.write(depfile)


if __name__ == "__main__":
    main()
