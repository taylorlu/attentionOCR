import os
import tensorflow as tf
import attention
import data_producer

def run_epoch(epoch, attention_model, producer, sess, save_path, saver):

    ops = [attention_model.cost, attention_model.learning_rate, attention_model.global_step, attention_model.train_op]

    for i in range(producer.get_batch_count()):
        inputs, labels, masks = producer.get_batch_sample()

        feed_dict = attention_model.feed_dict(inputs, labels, masks)
        cost, lr, step, _ = sess.run(ops, feed_dict)

        if(step%200==0):
            saver.save(sess, save_path)

        print('Epoch {}, iter {}: Cost= {:.2f}, lr= {:.2e}'.format(epoch, step, cost, lr))

    saver.save(sess, save_path)


def main(argv):
    restore_path = argv.get('restore_path', None)
    save_path = argv['save_path']
    producer = data_producer.DataProducer(argv['json_path'], argv['batch_size'], argv['max_step'])

    attention_model = attention.AttentionModel(argv)
    attention_model.init_inference_for_train()

    with tf.Session() as sess:
        restore_vars = []
        train_vars = []
        # for var in tf.global_variables():
        #     if(var.name.startswith('arcface/')):
        #         train_vars.append(var)
        #     else:
        #         if(not 'Adam' in var.name):
        #             restore_vars.append(var)

        attention_model.init_train()
        sess.run(tf.global_variables_initializer())

        if restore_path:
            saver = tf.train.Saver()
            saver.restore(sess, restore_path)
        saver = tf.train.Saver()

        print("Begin training...")
        for e in range(argv['epochs']):
            run_epoch(e, attention_model, producer, sess, save_path, saver)
            print("========"*5)
            print("Finished epoch", e)


if __name__=="__main__":

    args_params = {
        "save_path": r"saver/model.ckpt",
        "json_path": r"data.json",
        "restore_path": r"saver/model.ckpt",

        "batch_size": 64,
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
