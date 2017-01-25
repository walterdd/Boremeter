FROM boremeter/caffe-opencv

WORKDIR /root

# Clone AuditoryTracking repository
WORKDIR /root
RUN git clone https://github.com/walterdd/Boremeter.git && \
  cd Boremeter && python setup.py install
  
