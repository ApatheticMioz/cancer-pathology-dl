import re

with open("Research Report/output_humanized.tex", "r") as f:
    content = f.read()

content = content.replace(r"The shared en alongside $\lambda_{cls}$, allowing the model to learn balanced representations on its own $\lambda_{seg}$.",
                          r"The shared encoder alongside $\lambda_{cls}$, allowing the model to learn balanced representations on its own without a fixed $\lambda_{seg}$.")

content = content.replace("lightweight enbenefits", "lightweight encoder benefits")
content = content.replace("shared enupgrades", "shared encoder upgrades")
content = content.replace("shared en against", "shared encoder against")
content = content.replace("enprovides", "encoder provides")

with open("Research Report/output_humanized.tex", "w") as f:
    f.write(content)
