import os
import tensorflow as tf
import attention
import data_producer

def main(argv):
    restore_path = argv.get('restore_path', None)
    save_path = argv['save_path']
    producer = data_producer.DataProducer(argv['json_path'], argv['batch_size'], argv['max_step'])
    
    # from tensorflow.python import pywrap_tensorflow  
    # model_dir="saver/model.ckpt" 
    # reader = pywrap_tensorflow.NewCheckpointReader(model_dir)  
    # var_to_shape_map = reader.get_variable_to_shape_map()  
    # for key in var_to_shape_map:  
    #     print("tensor_name: ", key)
        #print(reader.get_tensor(key))
    # return
    attention_model = attention.AttentionModel(argv)
    attention_model.init_inference()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # for ele in tf.global_variables():
        #     # print(ele.name)
        #     if(ele.name[:-2] not in var_to_shape_map):
        #         print('===========', ele.name)
        if restore_path:
            saver = tf.train.Saver()
            saver.restore(sess, restore_path)
        
        inputs, labels, masks = producer.get_batch_sample()
        print(labels[30])

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            feed_dict = attention_model.feed_dict(inputs)
            predictions = sess.run(attention_model.predictions(), feed_dict)
            print(predictions)


if __name__=="__main__":

    args_params = {
        "save_path": r"saver/model.ckpt",
        "json_path": r"data.json",
        'restore_path': "saver/model.ckpt",

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