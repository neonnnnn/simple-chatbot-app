import json
import os
import sys

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

print("Transformers version", transformers.__version__)
set_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transformers_model_downloader(
    pretrained_model_name, do_lower_case, dir="./download"
):
    """This function, save the checkpoint, config file along with tokenizer config and vocab files
    of a transformer model of your choice.
    """
    print(f"Download model and tokenizer {pretrained_model_name}")
    # loading pre-trained model and tokenizer
    config = AutoConfig.from_pretrained(
        pretrained_model_name,
    )
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name, do_lower_case=do_lower_case
    )

    # NOTE : for demonstration purposes, we do not go through the fine-tune processing here.
    # A Fine_tunining process based on your needs can be added.
    # An example of  Fine_tuned model has been provided in the README.
    try:
        os.mkdir(dir)
    except OSError:
        print(f"Creation of directory {dir} failed")
    else:
        print(f"Successfully created directory {dir}")

    print(
        "Save model and tokenizer/ Torchscript model based on the setting from setup_config",
        pretrained_model_name,
        "in directory",
        dir,
    )

    model.save_pretrained(dir, safe_serialization=False)
    tokenizer.save_pretrained(dir, safe_serialization=False)

    return


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    if len(sys.argv) > 1:
        filename = os.path.join(dirname, sys.argv[1])
    else:
        filename = os.path.join(dirname, "setup_config.json")
    f = open(filename)
    settings = json.load(f)
    model_name = settings["model_name"]
    do_lower_case = settings["do_lower_case"]

    transformers_model_downloader(
        model_name,
        do_lower_case,
    )
