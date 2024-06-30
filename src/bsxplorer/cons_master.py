from .cons_utils import render_template, ConsoleScript, MetageneScript, CategoryScript, ChrLevelsScript, Path


def _save(script: ConsoleScript):
    html = script.main()
    savepath = script.args.dir / (script.args.out + ".html")

    package_path = Path(__file__).parent
    render_template(package_path / "html/MetageneTemplate.html", html, savepath)


def metagene(script: str = None):
    _save(MetageneScript(script.split()))


def category(script: str = None):
    _save(CategoryScript(script.split()))


def chr_levels(script: str = None):
    _save(ChrLevelsScript(script.split()))

