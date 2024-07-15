from .cons_utils import render_template, ConsoleScript, MetageneScript, CategoryScript, ChrLevelsScript, Path, BamScript


def _save(script: ConsoleScript):
    html = script.main()
    savepath = script.args.dir / (script.args.out + ".html")

    package_path = Path(__file__).parent
    render_template(package_path / "html/MetageneTemplate.html", html, savepath)


def metagene():
    _save(MetageneScript())


def category():
    _save(CategoryScript())


def chr_levels():
    _save(ChrLevelsScript())


def bam():
    script = BamScript()
    script.main()

