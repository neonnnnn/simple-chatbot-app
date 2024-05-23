FROM pytorch/torchserve:latest-cpu

WORKDIR /torchserve
COPY --chown=model-server . /torchserve 
USER model-server
RUN ./setup.sh
EXPOSE 8080
CMD ["torchserve", "--start", "--model-store", "./model_store", "--models", "my_tc=chatbot.mar", "--ncs"]