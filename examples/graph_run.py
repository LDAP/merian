#!/usr/bin/env python3
"""
Loads a graph config and runs it; without a config it opens an empty window.

  graph_run.py [options] [graph.json] [graph options] [args...]
  graph_run.py examples/hdr_viewer.json image.hdr

--help lists the options, with a config also the graph's.
"""

import argparse
import os
import signal
import sys

import lib  # noqa: F401
from lib.args import config_first

# isort: split
import merian
import merian.graph as mg


def _runner_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument("config", nargs="?", metavar="graph.json")
    parser.add_argument(
        "--loglevel", choices=["trace", "debug", "info", "warn", "error"]
    )
    parser.add_argument("--plugin-path", action="append", default=[], metavar="DIR")
    parser.add_argument(
        "--validation",
        choices=["on", "off", "ifdebug"],
        default="ifdebug",
        help="Vulkan validation layers (default: ifdebug)",
    )
    parser.add_argument(
        "--max-iterations", type=int, metavar="N", help="quit after N iterations"
    )
    parser.add_argument(
        "--exit-on-event",
        default="//end,//close",
        metavar="PATTERN",
        help="quit when a node sends a matching event, 'node type/identifier/event' "
        "(default: //end,//close, sent by a video that ran out, an image sink's exit "
        "trigger and a closed window; empty to stay)",
    )
    parser.add_argument(
        "--time-delta",
        type=float,
        metavar="MS",
        help="advance the graph time by a fixed delta per iteration instead of the wall clock",
    )
    parser.add_argument(
        "--print-times",
        action="store_true",
        help="profile, and print the report of the whole run and of the last graph build",
    )
    parser.add_argument(
        "--print-memory",
        action="store_true",
        help="print where the graph allocates memory",
    )
    return parser


def _default_description() -> mg.Description:
    description = mg.Description()
    description.add_node(
        "Window", "window", {"on close": {"send event": True, "remove node": False}}
    )
    description.add_node("ImGui", "imgui")
    description.add_connection("window", "imgui", "canvas", "canvas")
    description.add_connection("window", "imgui", "window", "window")
    return description


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    base = _runner_parser()
    argv = config_first(sys.argv[1:] if argv is None else argv, base)
    options, _ = base.parse_known_args(argv)
    if options.loglevel:
        merian.set_log_level(options.loglevel)
    for path in options.plugin_path:
        merian.plugins.add_search_path(path)

    try:
        if options.config is None:
            argparse.ArgumentParser(
                parents=[base],
                prog=base.prog,
                description=base.description,
                formatter_class=base.formatter_class,
            ).parse_args(argv)
            description, store_path = _default_description(), "merian-graph.json"
        else:
            config = mg.config.Config(options.config)
            mg.config.apply_args(config, argv, base)
            description = config.description()
            store_path = f"{config.path.stem}.stored.json"
    except mg.ConfigError as e:
        print(f"{base.prog}: {e}", file=sys.stderr)
        return 2

    validation = (
        merian.DEBUG_BUILD
        if options.validation == "ifdebug"
        else options.validation == "on"
    )
    context = merian.Context(
        extensions=[*mg.EXTENSIONS, merian.DEVICE_FAULT_EXTENSION],
        validation=validation,
        search_paths=mg.default_search_dirs(),
    )
    with mg.Graph(context, description, strict=False) as graph:
        graph.set_store_path(store_path)
        if options.time_delta is not None:
            graph.set_properties(
                {
                    "graph_properties": {
                        "time source": "Delta",
                        "delta": options.time_delta,
                    }
                }
            )
        graph.profiler.enabled = options.print_times

        # Also when started with SIGINT ignored (e.g. as a background job)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.default_int_handler)
        try:
            result = graph.run(
                iterations=options.max_iterations, until_event=options.exit_on_event
            )
            if result.stop_reason == mg.StopReason.EVENT:
                print(f"exiting on event {result.event}", file=sys.stderr)
        except KeyboardInterrupt:
            pass
        except merian.VulkanError as e:
            print(f"aborting on Vulkan error: {e}", file=sys.stderr)
            context.dump_device_fault()
            sys.stderr.flush()
            # Teardown after a device loss only fails again
            os._exit(1)
        graph.wait()

        if options.print_times:
            print(graph.profiler.report(), end="")
            print(f"Graph build:\n{graph.profiler.report(mg.ReportKind.BUILD)}", end="")
        if options.print_memory:
            print(graph.memory_report(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
