# -*- coding:UTF-8 -*-
# Autor: Rafael Viana
# Codigo: Arvore de Decision Distribuida ADD
import tensorflow as tf

cluster = tf.train.ClusterSpec({"local": ["svgov.ddns.net:3000","35.192.227.92:3001"]})

x = tf.constant(2)


with tf.device("/job:local/task:1"):
    y2 = x - 66

with tf.device("/job:local/task:0"):
    y1 = x + 300
    y = y1 + y2

with tf.Session("grpc://svgov.ddns.net:3001") as sess:
    result = sess.run(y)
    print(result)