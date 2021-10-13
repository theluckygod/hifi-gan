FROM nvcr.io/nvidia/pytorch:21.07-py3
USER root

COPY requirements.txt /tmp/requirements.txt
RUN pip install llvmlite --ignore-installed && \
    python3 -m pip install -r /tmp/requirements.txt

COPY . /hifi-gan
WORKDIR /hifi-gan

CMD /bin/bash