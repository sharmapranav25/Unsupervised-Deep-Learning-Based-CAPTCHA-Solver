"""Static triage of candidate CAPTCHA repositories in staging/.

Records, per repository: licence evidence, whether it contains its own image
rendering code, wrapper dependencies on other CAPTCHA packages, bundled fonts,
and keyword evidence for drawing effects. Keyword hits are leads for manual
review, not verified visible effects.
"""
import glob
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

EFFECTS = {
    "ttf_text": r"imagettftext|imagefttext|->text\(|annotateImage|annotate\(",
    "bitmap_text": r"imagestring\(|imagechar\(",
    "lines": r"imageline\(|->line\(|drawLine",
    "arcs_curves": r"imagearc\(|imagefilledarc|bezier|->arc\(|curve",
    "ellipses": r"imageellipse|imagefilledellipse|->ellipse\(|->circle\(",
    "dots_pixels": r"imagesetpixel\(|->pixel\(|noise",
    "rects_blocks": r"imagefilledrectangle\(|imagerectangle\(|->rectangle\(",
    "polygons": r"imagepolygon|imagefilledpolygon|->polygon\(",
    "wave_warp": r"wave|warp|distort|swirl|deform",
    "blur_filter": r"imagefilter\(|IMG_FILTER|blur|GaussianBlur|imageconvolution",
    "rotation": r"imagerotate|rotate|\$angle",
    "background_img": r"imagecreatefrom(png|jpeg|jpg|gif)",
    "gradient": r"gradient",
    "shadow": r"shadow",
    "gif_anim": r"GIFEncoder|animated|AnimatedGif",
    "svg": r"<svg|<path",
    "random_rgb": r"rand\(\s*0\s*,\s*255",
    "imagick": r"Imagick",
    "intervention": r"Intervention.Image",
}
WRAP = re.compile(
    r'"(gregwar/captcha|mews/captcha|dapphp/securimage|captcha-com/[\w-]+|'
    r'laminas/laminas-captcha|yiisoft/yii2|topthink/think-captcha|webman/captcha)"'
)
LICENCES = [
    ("MIT", r"Permission is hereby granted"),
    ("Apache-2.0", r"Apache License"),
    ("LGPL", r"LESSER GENERAL"),
    ("GPL-3", r"GNU GENERAL PUBLIC LICENSE\s+Version 3"),
    ("GPL-2", r"GNU GENERAL PUBLIC LICENSE\s+Version 2"),
    ("BSD-3", r"Neither the name"),
    ("BSD-2", r"Redistribution and use"),
    ("MPL", r"Mozilla"),
]


def norm(path):
    return path.replace(os.sep, "/").lower()


rows = []
for name in sorted(os.listdir("staging")):
    root = os.path.join("staging", name)
    src, files, lic, comp = "", 0, "?", ""
    paths = glob.glob(root + "/**/*", recursive=True)
    for p in paths:
        if not os.path.isfile(p):
            continue
        base = os.path.basename(p).lower()
        if base.startswith(("license", "licence", "copying")) and lic == "?":
            text = open(p, errors="ignore").read(4000)
            lic = next((k for k, pat in LICENCES if re.search(pat, text)), "other:" + base)
        if base == "composer.json" and not comp:
            comp = open(p, errors="ignore").read()
        n = norm(p)
        if n.endswith((".php", ".inc")) and "/vendor/" not in n and "/tests/" not in n:
            files += 1
            src += open(p, errors="ignore").read() + "\n"
    if lic == "?" and comp:
        m = re.search(r'"license"\s*:\s*\[?\s*"([^"]+)"', comp)
        lic = "composer:" + m.group(1) if m else "?"
    rows.append(
        dict(
            name=name,
            php_files=files,
            loc=src.count("\n"),
            license=lic,
            renders=bool(re.search(r"imagecreate|Imagick|Intervention.Image|<svg|Nette.Utils.Image", src, re.I)),
            wrapper=sorted(set(WRAP.findall(comp))),
            fonts=sum(norm(p).endswith((".ttf", ".otf", ".gdf")) for p in paths),
            effects=[k for k, pat in EFFECTS.items() if re.search(pat, src, re.I)],
        )
    )

json.dump(rows, open("analysis.json", "w"), indent=1)
for r in rows:
    print(
        f"{r['name'][:38]:38} loc={r['loc']:6} lic={r['license'][:14]:14} "
        f"rend={int(r['renders'])} fonts={r['fonts']:3} "
        f"wrap={','.join(r['wrapper'])[:28]:28} {' '.join(r['effects'])}"
    )
