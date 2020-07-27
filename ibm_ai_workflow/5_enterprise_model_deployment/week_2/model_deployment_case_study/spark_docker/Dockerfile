FROM jupyter/pyspark-notebook

RUN mkdir src && mkdir data

COPY src/. src/
COPY data/. data/

WORKDIR /src

CMD ["jupyter", \
    "notebook", \
    "--port=8888", \
    "--allow-root", \
    "--ip='0.0.0.0'" \
    "--no-browser",]
    # "--NotebookApp.token=''", \
    # "--NotebookApp.password=''"]