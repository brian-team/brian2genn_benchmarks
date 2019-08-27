FROM nvidia/cuda:8.0-devel

RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
   && curl -sSL https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
   && bash /tmp/miniconda.sh -bfp /usr/local \
   && rm -rf /tmp/miniconda.sh \
   && conda install -y python=2 \
   && conda update conda \
   && apt-get -qq -y remove curl bzip2 \
   && apt-get -qq -y autoremove \
   && apt-get autoclean \
   && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
   && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    gcc \
    make \
    g++ \
    cuda-core-8-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV CUDA_PATH=/usr/local/cuda
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}"

WORKDIR /root

RUN conda config --set restore_free_channel true
RUN conda install -c conda-forge --yes brian2=2.2
RUN pip install brian2genn==1.2
RUN git clone https://github.com/genn-team/genn
RUN (cd genn; git checkout 3.2.0)

ENV GENN_PATH=/root/genn

WORKDIR /root/brian2genn_benchmarks
