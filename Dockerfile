FROM nvidia/cuda:10.0-cudnn7-runtime
LABEL maintainer 'derushio'

USER root
ENV HOME /root

RUN apt update
RUN apt install -y git curl make gcc zlib1g-dev libssl-dev libffi-dev libbz2-dev libreadline-dev libsqlite3-dev

RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> $HOME/.bashrc
RUN echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> $HOME/.bashrc
RUN echo 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> $HOME/.bashrc

ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH
ENV PATH $PYENV_ROOT/shims:$PATH

RUN pyenv install 3.7.2
RUN pyenv global 3.7.2
RUN pip install -U pip
RUN pip install pipenv
