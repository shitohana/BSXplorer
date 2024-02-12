from .cons_utils import render_template, ConsoleScript, MetageneScript, CategoryScript, ChrLevelsScript


def _save(script: ConsoleScript):
    html = script.main()
    savepath = script.args.dir / (script.args.out + ".html")
    render_template("html/MetageneTemplate.html", html, savepath)


def metagene():
    _save(MetageneScript())


def category():
    _save(CategoryScript())


def chr_levels():
    _save(ChrLevelsScript())

