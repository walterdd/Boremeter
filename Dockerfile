FROM boremeter/caffe-opencv

WORKDIR /root

# Install pandas, Jinja2
RUN pip install update pip && \
  pip install Jinja2 pandas tqdm

# Clone AuditoryTracking repository
WORKDIR /root
RUN git clone https://github.com/walterdd/Boremeter.git && \
  cd Boremeter && python setup.py install

