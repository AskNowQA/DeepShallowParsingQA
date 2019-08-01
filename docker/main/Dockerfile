FROM python:3.7

RUN apt-get update
RUN apt-get install -y python-gevent python-gevent-websocket emacs
WORKDIR /data/workspace

RUN git clone https://github.com/AskNowQA/DeepShallowParsingQA
WORKDIR /data/workspace/DeepShallowParsingQA
RUN git checkout develop && git pull origin develop
COPY ./data/ /data/workspace/DeepShallowParsingQA/data/
RUN pip install --upgrade pip
RUN pip install Cython
#RUN pip install torchvision
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.1-cp37-cp37m-linux_x86_64.whl
RUN python -c 'import nltk; nltk.download("stopwords")'
ENV PYTHONPATH=.:$PYTHONPATH
RUN python scripts/web_api.py --port 5006 --checkpoint cp_LQ_str.chpt