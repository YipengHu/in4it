
import tensorflow as tf


def dice_loss(ps,ts,eps=1e-6):
    numerator = tf.reduce_sum(ts*ps,axis=[1,2,3])*2 + eps
    denominator = tf.reduce_sum(ts,axis=[1,2,3]) + tf.reduce_sum(ps,axis=[1,2,3]) + eps
    return 1 - numerator/denominator


def dice_binary(ps,ts,eps=1e-6):
    ps = tf.cast(ps>=.5,dtype=ps.dtype)
    ts = tf.cast(ts>=.5,dtype=ts.dtype)
    numerator = tf.reduce_sum(ts&ps,axis=[1,2,3])*2 + eps
    denominator = tf.reduce_sum(ts,axis=[1,2,3]) + tf.reduce_sum(ps,axis=[1,2,3]) + eps
    return 1 - numerator/denominator
