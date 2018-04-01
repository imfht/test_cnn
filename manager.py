#!/usr/bin/env python
# -*- coding: utf-8 -*-
from app import app
from gevent.wsgi import WSGIServer
http_server = WSGIServer(('', 5000), app.app)
http_server.serve_forever()

