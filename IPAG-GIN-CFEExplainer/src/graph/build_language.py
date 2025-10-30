import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
from tree_sitter import Language

class LanguageBuilder:
    def __init__(self, langs):
        self._language_map = {}
        self.langs = langs
    def _build_python(self):
        return tspython.language()
    def _build_c(self):
        return tsc.language()
    def _build_cpp(self):
        return tscpp.language()
    def _build_java(self):
        return tsjava.language()
    def build(self):
        for l in self.langs:
            match l:
                case "c":
                    self._language_map[l] = Language(self._build_c())
                case "py":
                    self._language_map[l] = Language(self._build_python())
                case "cpp":
                    self._language_map[l] = Language(self._build_cpp())
                case "java":
                    self._language_map[l] = Language(self._build_java())
        return self._language_map
    