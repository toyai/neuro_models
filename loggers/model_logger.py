from pytorch_lightning.core.memory import ModelSummary


def model_logger(name, network):
    with open(f"{name}.md", "w") as f:
        f.write(f"## {name}\n```py\n")
        f.write(str(network))
        f.write("\n```")


def model_summary_logger(name, ligtning):
    with open(f"{name}-summary.md", "w") as f:
        f.write(f"## {name}-summary\n```py\n")
        f.write(str(ModelSummary(ligtning, "full")))
        f.write("\n```")
