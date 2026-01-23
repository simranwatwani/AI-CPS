FROM busybox:latest

RUN mkdir -p /tmp/learningBase/train /tmp/learningBase/validation

COPY train/training_data.csv   /tmp/learningBase/train/
COPY validation/test_data.csv  /tmp/learningBase/validation/
COPY Readme.md                 /tmp/learningBase/

CMD ["tail", "-f", "/dev/null"]
