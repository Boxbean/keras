FROM tensorflow/tensorflow:latest-gpu

# RUN apt-get update
# RUN apt-get install -y libgl1-mesa-dev

RUN mkdir /workspace

WORKDIR /workspace
COPY . /workspace

RUN pip3 install keras numpy pillow
# opencv-python

CMD python3 mnist_mlp_v4.py
