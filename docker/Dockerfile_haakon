FROM nvcr.io/nvidia/pytorch:20.03-py3
RUN mkdir /pip_installs

COPY  requirements.txt /workspace

RUN pip install --upgrade pip
#RUN pip install -r requirements.txt
RUN pip install 'git+https://github.com/facebookresearch/fvcore'
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Install Detectron2
RUN git clone https://github.com/facebookresearch/detectron2 /detectron2_repo
ENV FORCE_CUDA="1"
RUN pip install -e /detectron2_repo
RUN pip install filterpy==1.4.5
#RUN pip install numba
RUN pip install scikit-image
RUN pip install lap
# Add densepose to python path

WORKDIR /workspace

RUN mkdir /pytorch_models
ENV TORCH_HOME=/workspace
RUN apt-get update
RUN apt install ffmpeg -y 