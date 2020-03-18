FROM nvidia/cuda:10.0-base-ubuntu16.04 AS deep-reco-gym

# Install some basic utilities
RUN apt-get update && apt-get install -y \
  curl \
  ca-certificates \
  sudo \
  git \
  bzip2 \
  libx11-6 \
  &&  rm -rf /var/lib/apt/lists/*

# JAVA
#ENV JAVA_VER 8
#ENV JAVA_HOME /usr/lib/jvm/java-8-oracle

# Install "software-properties-common" (for the "add-apt-repository")
RUN apt-get update && apt-get install -y \
  software-properties-common

# Add the "JAVA" ppa
RUN add-apt-repository -y \
  ppa:webupd8team/java

# Install OpenJDK-8
RUN apt-get update && \
  apt-get install -y openjdk-8-jdk && \
  apt-get install -y ant && \
  apt-get clean;

# Fix certificate issues
RUN apt-get update && \
  apt-get install ca-certificates-java && \
  apt-get clean && \
  update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME
# RUN echo 'deb http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main' >> /etc/apt/sources.list && \
#   echo 'deb-src http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main' >> /etc/apt/sources.list && \
#   apt-key adv --keyserver keyserver.ubuntu.com --recv-keys C2518248EEA14886 && \
#   apt-get update && \
#   echo oracle-java${JAVA_VER}-installer shared/accepted-oracle-license-v1-1 select true | sudo /usr/bin/debconf-set-selections && \
#   apt-get install -y --force-yes --no-install-recommends oracle-java${JAVA_VER}-installer oracle-java${JAVA_VER}-set-default && \
#   apt-get clean && \
#   rm -rf /var/cache/oracle-jdk${JAVA_VER}-installer

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
  && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
  && chmod +x ~/miniconda.sh \
  && ~/miniconda.sh -b -p ~/miniconda \
  && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda create -y --name py36 python=3.6.9 \
  && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN /home/user/miniconda/bin/conda install conda-build=3.18.9=py36_3 \
  && /home/user/miniconda/bin/conda clean -ya

# CUDA 10.0-specific steps
# RUN conda install -y -c pytorch \
#   cudatoolkit=10.0 \
#   "pytorch=1.2.0=py3.6_cuda10.0.130_cudnn7.6.2_0" \
#   "torchvision=0.4.0=py36_cu100" \
#   && conda clean -ya

ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH




# Add Project
COPY recommendation /app/recommendation
COPY output/trivago /app/output/trivago
COPY environment.yml /app
COPY luigi.cfg /app
COPY taskrunner.sh /app

RUN sudo chown -R user:user /app

# Set the default command to python3
EXPOSE 8501

#ADD ./luigi/taskrunner.sh /luigi/
#ENTRYPOINT ["bash", "/luigi/taskrunner.sh"]
ENV CONDA_DEFAULT_ENV=deep-reco-gym
ENV OUTPUT_PATH=/app/output

#CMD ["/bin/bash"]
ENTRYPOINT ["bash", "/app/taskrunner.sh"]
