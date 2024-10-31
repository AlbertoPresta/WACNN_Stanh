FROM gitlab.di.unito.it:5000/presta/resdsic


RUN pip install compressai 
RUN  pip install torchac
RUN pip install ipywidgets
RUN pip install Ninja
RUN pip install psutil
RUN pip install pytest-gc
RUN pip install timm
RUN apt update -y
RUN apt install -y gcc
RUN apt install -y g++ 
RUN pip install einops
RUN  pip install seaborn

WORKDIR /src
COPY src /src 



RUN chmod 775 /src
RUN chown -R :1337 /src

ENTRYPOINT [ "python3"]
