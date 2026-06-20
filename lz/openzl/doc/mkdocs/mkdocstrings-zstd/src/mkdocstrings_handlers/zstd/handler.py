import os
from collections import ChainMap
from pathlib import Path
from typing import Any, ClassVar, List, Mapping, Optional, Tuple

from markdown import Markdown
from mkdocstrings.handlers.base import BaseHandler, CollectionError

from .doxygen import Doxygen, DoxygenObject
from .rendering import AutorefsHook, do_format


class ZstdHandler(BaseHandler):
    default_config: ClassVar[Mapping[str, Any]] = {
        "show_description": True,
        "show_description_preconditions": True,
        "show_description_postconditions": True,
        "show_description_parameters": True,
        "show_description_exceptions": True,
        "show_description_return": True,
        "show_description_parameter_type": False,
        "show_symbol_type_toc": True,
        "show_signature": True,
        "show_root_toc_entry": True,
        "show_compound_toc_entries": True,
        "show_compound_root_heading": True,
        "show_enum_toc_entries": True,
        "show_function_toc_entries": True,
        "show_define_toc_entries": True,
        "show_typedef_toc_entries": True,
        "show_variable_toc_entries": False,
        "show_define_initializer": True,
        "clang_format_based_on_style": "InheritParentConfig",
        "heading_level": 2,
        "line_length": 80,
    }

    def __init__(
        self,
        *args: Any,
        source_directory: Path,
        include_paths: List[Path],
        build_directory: Path,
        sources: List[Path],
        predefined: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Ensure we have an absolute paths
        source_directory = os.path.abspath(source_directory)
        include_paths = [os.path.abspath(d) for d in include_paths]
        build_directory = os.path.abspath(build_directory)

        self.config = {
            "source_directory": source_directory,
            "include_paths": include_paths,
        }

        self._doxygen = Doxygen(
            source_directory=source_directory,
            include_paths=include_paths,
            sources=sources,
            xml_output=os.path.join(build_directory, "doxyxml"),
            predefined=predefined or [],
        )

    def collect(self, identifier: str, _config: Mapping[str, Any]) -> DoxygenObject:
        obj = self._doxygen.collect(identifier)
        if obj is None:
            raise CollectionError(f"Failed to find {identifier}")
        return obj

    def get_anchors(self, obj: DoxygenObject) -> Tuple[str]:
        # TODO: Typedefs?
        return (obj.qualified_name,)

    def update_env(self, md: Markdown, config: dict) -> None:
        """Update the Jinja environment with custom filters and tests.

        Parameters:
            md: The Markdown instance.
            config: The configuration dictionary.
        """
        super().update_env(md, config)
        self.env.trim_blocks = True
        self.env.lstrip_blocks = True
        self.env.keep_trailing_newline = False
        self.env.filters["format"] = do_format
        self.env.globals["AutorefsHook"] = AutorefsHook

    def render(self, obj: DoxygenObject, config: dict) -> str:
        final_config = ChainMap(config, self.config, self.default_config)

        template_name = f"{obj.kind}.html.jinja"
        template = self.env.get_template(template_name)

        heading_level = final_config["heading_level"]

        return template.render(
            **{
                "config": final_config,
                "heading_level": heading_level,
                "root": True,
                obj.kind: obj,
            }
        )


def get_handler(
    theme: str,
    custom_templates: Optional[str] = None,
    config_file_path: Optional[Path] = None,
    source_directory: Optional[Path] = None,
    include_paths: Optional[List[Path]] = None,
    build_directory: Optional[Path] = None,
    sources: Optional[List[Path]] = None,
    predefined: Optional[List[str]] = None,
    **_config: Any,
) -> ZstdHandler:
    """Return an instance of `ZstdHandler`.

    Arguments:
        theme: The theme to use when rendering contents.
        custom_templates: Directory containing custom templates.
        config_file_path: Path to the mkdocs configuration file.
        source_directory: Directory containing the source files relative to
                          directory of the config_file_path.
        include_paths: List of directories to include in the search path relative
                       to the source directory.
        build_directory: Directory where the Doxygen files will be generated
                         relative to directory of the config_file_path.
        sources: List of source files to process.
        **_config: Configuration passed to the handler.
    """
    if config_file_path is None:
        raise ValueError("config_file_path is required")
    if source_directory is None:
        raise ValueError("source_directory is required")
    if include_paths is None:
        include_paths = []
    if build_directory is None:
        raise ValueError("build_directory is required")
    if sources is None:
        raise ValueError("sources is required")

    config_file_dir = os.path.dirname(config_file_path)
    source_directory = os.path.join(config_file_dir, source_directory)
    include_paths = [os.path.join(source_directory, d) for d in include_paths]
    build_directory = os.path.join(config_file_dir, build_directory)

    return ZstdHandler(
        handler="zstd",
        source_directory=source_directory,
        include_paths=include_paths,
        build_directory=build_directory,
        sources=sources,
        predefined=predefined,
        theme=theme,
        custom_templates=custom_templates,
    )
