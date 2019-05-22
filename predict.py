import numpy as np
import os
import cv2
import tensorflow as tf
import attention
import data_producer
from misc import CaptionData, TopN


def main(argv):
    restore_path = argv.get('restore_path', None)
    save_path = argv['save_path']
    # producer = data_producer.DataProducer(argv['json_path'], argv['batch_size'], argv['max_step'])
    
    attention_model = attention.AttentionModel(argv)
    _probs, _last_output, _last_memory = attention_model.init_inference()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        if restore_path:
            saver = tf.train.Saver()
            saver.restore(sess, restore_path)

        batch_size = 1
        beam_size = 3
        max_caption_length = 40
        images = cv2.imread('images/test1.jpg')

        feed_dict = attention_model.feed_dict([images])
        initial_memory, initial_output = sess.run([attention_model.initial_memory, attention_model.initial_output], feed_dict)

        partial_caption_data = []
        complete_caption_data = []
        for k in range(batch_size):
            initial_beam = CaptionData(sentence = [],
                                       memory = initial_memory[k],
                                       output = initial_output[k],
                                       score = 1.0)
            partial_caption_data.append(TopN(beam_size))
            partial_caption_data[-1].push(initial_beam)
            complete_caption_data.append(TopN(beam_size))

        # Run beam search
        for idx in range(max_caption_length):
            partial_caption_data_lists = []
            for k in range(batch_size):
                data = partial_caption_data[k].extract()
                partial_caption_data_lists.append(data)
                partial_caption_data[k].reset()

            num_steps = 1 if idx == 0 else beam_size
            for b in range(num_steps):
                if idx == 0:
                    last_word = np.zeros((batch_size), np.int32)
                else:
                    last_word = np.array([pcl[b].sentence[-1]
                                        for pcl in partial_caption_data_lists],
                                        np.int32)

                last_memory = np.array([pcl[b].memory
                                        for pcl in partial_caption_data_lists],
                                        np.float32)
                last_output = np.array([pcl[b].output
                                        for pcl in partial_caption_data_lists],
                                        np.float32)

                feed_dict = attention_model.feed_dict([images],
                                                    last_word=last_word,
                                                    last_memory=last_memory,
                                                    last_output=last_output)
                scores, output, memory = sess.run([_probs, _last_output, _last_memory], feed_dict)

                # Find the beam_size most probable next words
                for k in range(batch_size):
                    caption_data = partial_caption_data_lists[k][b]
                    words_and_scores = list(enumerate(scores[k]))
                    words_and_scores.sort(key=lambda x: -x[1])
                    words_and_scores = words_and_scores[0:beam_size+1]

                    # Append each of these words to the current partial caption
                    for w, s in words_and_scores:
                        sentence = caption_data.sentence + [w]
                        score = caption_data.score * s
                        beam = CaptionData(sentence,
                                           memory[k],
                                           output[k],
                                           score)
                        if(w==3581):
                            complete_caption_data[k].push(beam)
                        else:
                            partial_caption_data[k].push(beam)

        results = []
        for k in range(batch_size):
            if complete_caption_data[k].size() == 0:
                complete_caption_data[k] = partial_caption_data[k]
            results.append(complete_caption_data[k].extract(sort=True))

        for r in results:
            for i in r:
                print(i.sentence)
        return results



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
