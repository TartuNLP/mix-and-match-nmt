from fairseq.data.multilingual.multilingual_data_manager import MultilingualDatasetManager, SRC_DICT_NAME, TGT_DICT_NAME
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

@register_task("mix_and_match_multilingual_translation")
class MixAndMatchMultilingualTranslationTask(TranslationMultiSimpleEpochTask):
    @staticmethod
    def add_args(parser):
        TranslationMultiSimpleEpochTask.add_args(parser)
        parser.add_argument('--disable-src-augmentation', action='store_true')
        parser.add_argument('--disable-tgt-augmentation', action='store_true')

    @classmethod
    def setup_task(cls, args, **kwargs):
        langs, dicts, training = MultilingualDatasetManager.prepare(
            cls.load_dictionary, args, **kwargs
        )

        if args.disable_src_augmentation:
            assert args.source_dict is not None
            dicts[SRC_DICT_NAME] = cls.load_dictionary(args.source_dict)

        if args.disable_tgt_augmentation:
            assert args.target_dict is not None
            dicts[TGT_DICT_NAME] = cls.load_dictionary(args.target_dict)

        return cls(args, langs, dicts, training)