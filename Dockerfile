FROM boremeter/caffe-opencv

# Clone AuditoryTracking repository
WORKDIR /root
RUN git clone https://github.com/walterdd/Boremeter.git && \
  cd Boremeter && python setup.py install
  
