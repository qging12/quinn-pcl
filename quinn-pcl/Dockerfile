# docker build -t ubuntu1804py36
FROM mawall/ubuntu16.04_base

# Packages
RUN apt-get update && apt-get install -y libpcl-dev=1.7.2-14build1
RUN pip install -U setuptools && \
    pip install         \
        python-pcl

RUN apt-get update && \
        apt-get install -y software-properties-common vim 


#RUN add-apt-repository ppa:jonathonf/python-3.6

RUN apt-get update -y

RUN apt-get install cmake -y && \
    #apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv && \
    apt-get install -y git && \
    apt-get install openni2-utils -y
# && \
#    apt-get install libpcl-dev -y

# fork module
#RUN git clone -b rc_patches4 https://github.com/Sirokujira/python-pcl.git
RUN git clone https://github.com/qging12/quinn-pcl.git
# main
# RUN git clone -b master https://github.com/strawlab/python-pcl.git

WORKDIR /quinn-pcl/python-pcl

# update pip
RUN python -m pip install pip --upgrade && \
    python -m pip install wheel

RUN pip install -r requirements.txt && \
    python setup.py build_ext -i && \
    python setup.py install

WORKDIR /quinn-pcl



