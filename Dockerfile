FROM python:3.7

RUN /usr/local/bin/python -m pip install --upgrade pip

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install -r requirements.txt
COPY . /opt/app

RUN pip install torch
RUN pip install rdkit-pypi
RUN pip install -r requirements.txt
COPY . /opt/app

USER $MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/app

WORKDIR /opt/app
RUN /usr/local/bin/python -m pip install -e .
