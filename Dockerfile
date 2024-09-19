FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# `tzdata` requires noninteractive mode.
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.9
ARG APP_DIR="/app"
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Tokyo \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    # pip:
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    # poetry:
    POETRY_NO_INTERACTION=1 \
    DEB_PYTHON_INSTALL_LAYOUT=deb \
    APP_DIR=${APP_DIR} \
    PYTHONPATH=${APP_DIR}/src:$PYTHONPATH
# Install system dependencies
RUN apt update \
 && apt install -y --no-install-recommends \
    curl \
    wget \
    software-properties-common
# Install python and poetry
RUN add-apt-repository ppa:deadsnakes/ppa \
 && apt update \
 && apt install -y python${PYTHON_VERSION}-dev python3-pip python${PYTHON_VERSION}-distutils \
 && apt clean \
 && rm -rf /var/lib/apt/lists/* \
 && unlink /usr/bin/python3 \
 && ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python3
# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 - -yfp \
 && cd /usr/local/bin \
 && ln -s /root/.local/bin/poetry \
 && poetry config virtualenvs.create false \
 && poetry config installer.max-workers 10

WORKDIR ${APP_DIR}
COPY pyproject.toml poetry.lock ./

RUN poetry export --without-hashes --no-interaction --no-ansi -f requirements.txt -o requirements.txt \
 && pip3 install --force-reinstall -r requirements.txt

COPY . .

CMD ["/bin/bash"]