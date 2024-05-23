# simple-chatbot-app

An implementation of a simple chatbot application in Python with PyTorch, TorchServe, Hugging Face, and streamlit.

## Requirements

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Usage

Exec the fllowing commands:

```bash
git clone git@github.com:neonnnnn/simple-chatbot-app.git
cd simple-chatbot-app
docker compose build
docker compose up
```

Then, the simple chatbot app runs on <http://localhost:8501/>.

You can change the underlying LLM by fixing `"model_name"` in `torchserve/setup_confing.json` (additionaly it might be required to fix the `setup.sh` (the model archiving part)).

## References

- [Build a bsic LLM chat app](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)
- [Serving Huggingface Transformers using TorchServe](https://github.com/pytorch/serve/tree/master/examples/Huggingface_Transformers)
