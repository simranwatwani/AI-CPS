FROM busybox:latest

RUN mkdir -p /app/knowledgeBase

COPY currentAiSolution.pkl /app/knowledgeBase/
COPY currentOlsSolution.pkl /app/knowledgeBase/
COPY scaler.pkl /app/knowledgeBase/
COPY Readme.md /app/knowledgeBase/

CMD ["tail", "-f", "/dev/null"]
