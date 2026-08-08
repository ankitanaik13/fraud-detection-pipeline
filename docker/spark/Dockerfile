FROM apache/spark:3.5.9-python3

# applyInPandasWithState (used by src/streaming/fraud_detector.py for per-account rolling
# state) needs pandas + pyarrow, which the base image doesn't bundle.
USER root
RUN pip3 install --no-cache-dir pandas==2.2.3 pyarrow==17.0.0

# Base image has no passwd entry for the spark user, so $HOME is /nonexistent — give
# spark-submit's --packages a writable Ivy cache dir instead (volume-mounted for reuse).
RUN mkdir -p /opt/spark-ivy && chown -R spark:spark /opt/spark-ivy
ENV HOME=/opt/spark-ivy
USER spark
