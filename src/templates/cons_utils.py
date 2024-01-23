from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template


def render_template(
        template: str | Path,
        params: dict,
        output: str | Path
):
    with open(template) as template_file:
        j2_template = Template(template_file.read())

        with open(output, "w") as output_file:
            output_file.write(
                j2_template.render(params)
            )


@dataclass
class TemplateMetagenePlot:
    title: str
    data: str


@dataclass
class TemplateMetageneContext:
    heading: str
    caption: str
    plots: list[TemplateMetagenePlot]


@dataclass
class TemplateMetageneBody:
    title: str
    context_reports: list[TemplateMetageneContext]