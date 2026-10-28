"""Restrict wheel modules to the same explicit file set used by source builds."""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.errors import SetupError


class PublicBuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        root = Path(__file__).resolve().parent
        allowed = set((root / "public-files.txt").read_text().splitlines())
        return [
            item
            for item in super().find_package_modules(package, package_dir)
            if str(Path(item[2]).resolve().relative_to(root)) in allowed
        ]

    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        expected = {
            name.removeprefix("python/")
            for name in (root / "public-files.txt").read_text().splitlines()
            if name.startswith("python/asterion/")
        }
        for path in Path(self.build_lib).rglob("*"):
            if path.is_file() and str(path.relative_to(self.build_lib)) not in expected:
                raise SetupError(
                    "Unlisted file in build directory; build from a fresh public preview"
                )


if __name__ == "__main__":
    setup(cmdclass={"build_py": PublicBuildPy})
