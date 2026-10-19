import json
import os
import tempfile
import unittest
from pathlib import Path

import merian.graph as mg


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def write(self, name: str, content: dict) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))
        return path

    def compose(self, config: dict, args: list[str]) -> dict:
        composed = mg.config.Config(self.write("graph.json", config), [])
        mg.config.apply_args(composed, args)
        return composed.json

    def presets(self) -> dict:
        self.write(
            "renderers/pt.json", {"order": ["pt"], "cli": {"spp": {"pointer": "/spp"}}}
        )
        self.write("renderers/fast.json", {"order": ["fast"]})
        self.write("mixins/debug.json", {"$+$order": ["debug"]})
        self.write(
            "mixins/filtered.json", {"include": "debug.json", "$+$order": ["filtered"]}
        )
        self.write(
            "mixins/fireflies.json",
            {"include": "debug.json", "$+$order": ["fireflies"]},
        )
        return {
            "spp": 1,
            "presets": {
                "renderer": {"folder": "renderers", "default": "pt", "order": 0},
                "mixin": {"folder": "mixins", "multiple": True, "order": 1},
            },
        }

    def test_alias_with_set(self):
        config = {
            "env": {"type": "Empty", "path": ""},
            "cli": {
                "env-map": {"pointer": "/env/path", "set": {"/env/type": "LatLong"}}
            },
        }
        self.assertEqual(
            self.compose(config, ["--env-map", "sky.hdr"])["env"],
            {"type": "LatLong", "path": "sky.hdr"},
        )

    def test_value_is_typed_like_the_current_one(self):
        config = {
            "v": 0,
            "s": "",
            "cli": {"v": {"pointer": "/v"}, "s": {"pointer": "/s"}},
        }
        composed = self.compose(config, ["--v", "8", "--s", "8"])
        self.assertEqual((composed["v"], composed["s"]), (8, "8"))

    def test_positional_arguments(self):
        config = {"args": "", "cli": {"--": {"pointer": "/args"}}}
        self.assertEqual(self.compose(config, ["scene.glb"])["args"], "scene.glb")
        self.assertEqual(
            self.compose(config, ["-heapsize", "64", "+map", "a b"])["args"],
            '-heapsize 64 +map "a b"',
        )
        self.assertEqual(self.compose(config, ["--", "+set a b"])["args"], "+set a b")
        self.assertEqual(self.compose(config, ["--", "--literal"])["args"], "--literal")

    def test_unknown_option(self):
        with self.assertRaises(mg.ConfigError):
            self.compose(
                {"args": "", "cli": {"--": {"pointer": "/args"}}}, ["--nope", "x"]
            )
        with self.assertRaises(mg.ConfigError):
            self.compose({}, ["x"])

    def test_required_is_satisfied_by_a_value(self):
        config = {"file": "", "cli": {"--": {"pointer": "/file", "required": True}}}
        with self.assertRaises(mg.ConfigError):
            self.compose(config, [])
        config["file"] = "stored.glb"
        self.assertEqual(self.compose(config, [])["file"], "stored.glb")

    def test_presets_apply_in_their_order(self):
        config = self.presets()
        self.assertEqual(self.compose(config, [])["order"], ["pt"])
        self.assertEqual(self.compose(config, ["--spp", "4"])["spp"], 4)
        self.assertEqual(
            self.compose(config, ["--mixin", "debug", "--renderer", "fast"])["order"],
            ["fast", "debug"],
        )
        with self.assertRaises(mg.ConfigError):
            self.compose(config, ["--renderer", "pt", "--renderer", "fast"])

    def test_includes_apply_once(self):
        composed = self.compose(
            self.presets(), ["--mixin", "filtered", "--mixin", "fireflies"]
        )
        self.assertEqual(composed["order"], ["pt", "debug", "filtered", "fireflies"])
        self.write("loop.json", {"include": "loop.json", "v": 1})
        self.assertEqual(self.compose({"include": "loop.json"}, [])["v"], 1)

    def test_preset_custom_file(self):
        self.write("renderers/pt.json", {"r": "pt"})
        custom = self.write("elsewhere/custom.json", {"r": "custom"})
        config = {"presets": {"renderer": {"folder": "renderers", "default": "pt"}}}
        self.assertEqual(
            self.compose(config, ["--renderer", str(custom)])["r"], "custom"
        )

    def test_include_before_the_content(self):
        self.write("parts/base.json", {"a": {"x": 1, "y": 1}, "list": [1]})
        fragment = self.write(
            "fragment.json",
            {"include": "parts/base.json", "a": {"y": 2}, "$+$list": [2]},
        )
        composed = self.compose({"list": [0]}, ["--merge", str(fragment)])
        self.assertEqual((composed["a"], composed["list"]), ({"x": 1, "y": 2}, [1, 2]))

    def test_working_directory_does_not_shadow(self):
        self.write("renderers/pt.json", {"r": "config"})
        decoy = Path(tempfile.mkdtemp(dir=self.root))
        (decoy / "renderers").mkdir()
        (decoy / "renderers" / "pt.json").write_text(json.dumps({"r": "cwd"}))
        config = {"presets": {"renderer": {"folder": "renderers", "default": "pt"}}}
        cwd = os.getcwd()
        os.chdir(decoy)
        try:
            self.assertEqual(self.compose(config, [])["r"], "config")
        finally:
            os.chdir(cwd)

    def test_null_removes_alias(self):
        self.write("renderers/di.json", {"cli": {"spp": None}})
        config = {
            "cli": {"spp": {"pointer": "/spp"}},
            "presets": {"renderer": {"folder": "renderers"}},
        }
        with self.assertRaises(mg.ConfigError):
            self.compose(config, ["--renderer", "di", "--spp", "2"])

    def test_alias_without_pointer(self):
        with self.assertRaises(mg.ConfigError):
            self.compose({"cli": {"renderer": {"help": "x"}}}, [])

    def test_alias_collision(self):
        with self.assertRaises(mg.ConfigError):
            self.compose({"cli": {"set": {"pointer": "/x"}}}, [])

    def test_set(self):
        composed = self.compose(
            {"a": {"list": [0, 0]}}, ["--set", "/a/b~1c=1", "--set", "/a/list/1=2"]
        )
        self.assertEqual(composed["a"], {"list": [0, 2], "b/c": 1})

    def test_set_appends_or_raises(self):
        composed = self.compose(
            {"list": [0]}, ["--set", "/list/1=1", "--set", "/list/-=2"]
        )
        self.assertEqual(composed["list"], [0, 1, 2])
        for pointer in ("/list/3", "/list/x", "/value/x"):
            with self.subTest(pointer), self.assertRaises(mg.ConfigError):
                self.compose({"list": [0], "value": 1}, ["--set", f"{pointer}=1"])


if __name__ == "__main__":
    unittest.main()
