FROM python:3.7

RUN apt-get update
RUN apt-get install -y python-gevent python-gevent-websocket emacs
WORKDIR /data/hamid/workspace

RUN git clone https://github.com/AskNowQA/DeepShallowParsingQA
WORKDIR /data/hamid/workspace/DeepShallowParsingQA
RUN git checkout develop && git pull origin develop
COPY ./data /data/hamid/workspace/DeepShallowParsingQA/data
RUN pip install --upgrade pip
RUN pip install Cython
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.1-cp37-cp37m-linux_x86_64.whl
RUN pip install torchvision
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c 'import nltk; nltk.download("stopwords")'
ENV PYTHONPATH=.:$PYTHONPATH