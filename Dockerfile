FROM ubuntu
ADD ./ /app
WORKDIR /app
RUN apt update&&apt install python3 python3-pip -y
RUN pip3 install -r requirements.txt
CMD ["python3","manager.py"]
