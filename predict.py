import numpy as np
import os
import cv2
import tensorflow as tf
import attention


def main(argv):
    restore_path = argv.get('restore_path', None)
    
    attention_model = attention.AttentionModel(argv)
    predictions = attention_model.init_inference()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        if restore_path:
            saver = tf.train.Saver()
            saver.restore(sess, restore_path)

        images = cv2.imread('images/test1.jpg')

        feed_dict = attention_model.feed_dict([images])
        pred = sess.run(predictions, feed_dict)
        print(pred)


if __name__=="__main__":

    args_params = {
        "save_path": r"saver/model.ckpt",
        "json_path": r"data.json",
        'restore_path': "saver/model.ckpt",

        "batch_size": 1,
        "epochs": 1000,
        "learning_rate": 0.002,
        "max_grad_norm": 100,
        "decay_steps": 500,
        "decay_rate": 0.95,

        "num_lstm_units": 256,
        "max_step": 50,
        "embedding_dim": 512,
        "vocabulary_size": 3582
    }
    main(args_params)
