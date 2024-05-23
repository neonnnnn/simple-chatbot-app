import ast
import json
import logging
import os
from abc import ABC

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info(f"Transformers version {transformers.__version__}")


class TransformersConversationHandler(BaseHandler, ABC):
    """
    Transformers handler class for conversation.
    """

    def __init__(self):
        super(TransformersConversationHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the model is loaded and initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        # read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the setup_config.json file.")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True
        )

        self.model.to(self.device)
        if any(
            fname
            for fname in os.listdir(model_dir)
            if fname.startswith("vocab.") and os.path.isfile(fname)
        ):
            logger.info("vocab file exists.")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                do_lower_case=self.setup_config["do_lower_case"],
                local_files_only=True,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.setup_config["model_name"],
                do_lower_case=self.setup_config["do_lower_case"],
            )
        self.model.eval()
        logger.info(f"Transformer model from path {model_dir} loaded successfully")

        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=self.setup_config.get("max_length", 30),
            do_sample=self.setup_config.get("do_sample", False),
            num_beams=self.setup_config.get("num_beams", 1),
            early_stopping=self.setup_config.get("early_stopping", True),
        )
        self.text_inputs = [
            {"role": "user", "content": "From now your name is Hyakumanben-bot."}
        ]
        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        input_texts = []
        for data in requests:
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            input_text_target = ast.literal_eval(input_text)
            input_text = input_text_target["text"]
            logger.info(f"Received text: '{input_text}'")
            input_texts.append(input_text)
        return input_texts

    def inference(self, input_texts):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        if self.initialized:
            for input_text in input_texts:
                self.text_inputs.append({"role": "user", "content": input_text})
            results = self.pipeline(
                self.text_inputs,
            )
            return [results[-1]["generated_text"][-1]["content"]]
        else:
            return ["Now initializing. Please try again later."]

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output
