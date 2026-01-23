FROM busybox:latest

RUN mkdir -p /app/activationBase

COPY activation_data.csv /app/activationBase/
COPY Readme.md /app/activationBase/

CMD ["tail", "-f", "/dev/null"]
