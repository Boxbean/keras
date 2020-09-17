FROM tensorflow/tensorflow:latest-gpu

RUN mkdir /workspace

WORKDIR /workspace
COPY . /workspace

RUN pip install keras

CMD python3 mnist_hierarchical_rnn.py
