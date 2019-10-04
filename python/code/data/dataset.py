class ReaderManager(object):

    def run(self, context, data_config):
        reader = context['reader']

        text_path = data_config['text_path']

        print('Reading text: {}'.format(text_path))
        reader_result = reader.read(text_path)
        sentences = reader_result['sentences']
        extra = reader_result['extra']
        metadata = reader_result.get('metadata', {})
        logger.info('len(sentences)={}'.format(len(sentences)))
        if 'n_etypes' in metadata:
            logger.info('n_etypes={}'.format(metadata['n_etypes']))

        word2idx = build_text_vocab(sentences)
        logger.info('len(vocab)={}'.format(len(word2idx)))

        if 'embeddings' in metadata:
            logger.info('Using embeddings from metadata.')
            embeddings = metadata['embeddings']
            del metadata['embeddings']
        else:
            logger.info('Reading embeddings.')
            embeddings, word2idx = EmbeddingsReader().get_embeddings(
                options, embeddings_path, word2idx)

        unk_index = word2idx.get(UNK_TOKEN, None)
        logger.info('Converting tokens to indexes (unk_index={}).'.format(unk_index))
        sentences = indexify(sentences, word2idx, unk_index)

        return {
            "sentences": sentences,
            "embeddings": embeddings,
            "word2idx": word2idx,
            "extra": extra,
            "metadata": metadata,
        }