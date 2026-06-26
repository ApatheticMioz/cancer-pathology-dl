import re

with open("Research Report/output_humanized.tex", "r") as f:
    content = f.read()

content = content.replace("baselinies", "baselines")
content = content.replace("Tao et al., 2\n\n21", "Tao et al., 2021")
content = content.replace("enprovides", "encoder provides")
content = content.replace("history datasets", "histology datasets")
content = content.replace("shared UNet enor", "shared UNet encoder or")
content = content.replace("task  on the shared en defines the relative inverse training rate. The process minimizes: $i$ $r_i$", "task $i$ on the shared encoder, and $r_i$ defines the relative inverse training rate. The process minimizes:")
content = content.replace("shared en against", "shared encoder against")
content = content.replace("shared enupgrades", "shared encoder upgrades")
content = content.replace("under V1,.1 achieves", "under V1. V2.1 achieves")
content = content.replace("enbenefits", "encoder benefits")
content = content.replace("(Varadarajan et al., 20; Pafka et al., 222)", "(Varadarajan et al., 2020; Pafka et al., 2022)")
content = content.replace("The shared en alongside $\lambda_{cls}$, allowing the model to learn balanced representations on its own $\lambda_{seg}$.", "The shared encoder alongside $\lambda_{cls}$, allowing the model to learn balanced representations on its own without a fixed $\lambda_{seg}$.")

with open("Research Report/output_humanized.tex", "w") as f:
    f.write(content)
