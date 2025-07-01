import os
from huggingface_hub import InferenceClient

hf = InferenceClient(token=os.getenv('HF_TOKEN'))


def load_env():
    import dotenv
    dotenv.load_dotenv()


def get_hosted_models(tag):
    from huggingface_hub import HfApi
    api = HfApi()
    models_generator = api.list_models(pipeline_tag=tag, inference="warm")
    return [model.modelId for model in models_generator]


def get_all_models(tag, model_filter=''):
    from huggingface_hub import HfApi
    api = HfApi()
    models_generator = api.list_models(pipeline_tag=tag)
    models = list(models_generator)
    return [model.modelId for model in models if model_filter in model.modelId]


def sentence_similarity(model, value1, value2):
    return hf.sentence_similarity(value1, [value2], model=model)


def feature_extraction(model, value):
    return hf.feature_extraction(value, model=model)


if __name__ == '__main__':
    load_env()

    all_fe_models = get_all_models('feature-extraction')
    print(f'Count of All Feature Extraction Models: {len(all_fe_models)}')

    all_fe_models_filtered = get_all_models('feature-extraction', 'intfloat/')
    print(f'Count of All Feature Extraction Models Filtered: {len(all_fe_models_filtered)}')

    all_hosted_fe_models = get_hosted_models('feature-extraction')
    print(f'Count All Hosted Feature Extraction Models: {len(all_hosted_fe_models)}')

    all_hosted_ss_models = get_hosted_models('sentence-similarity')
    print(f'Count All Hosted Sentence Similarity Models: {len(all_hosted_ss_models)}')

    _feature_extraction = feature_extraction('intfloat/multilingual-e5-large', 'Who are you?.')
    print(f'Feature Extraction: {_feature_extraction}')

    _sentence_similarity = sentence_similarity('intfloat/multilingual-e5-small', 'Who are you?', 'Who are you?')
    print(f'Sentence Similarity: {_sentence_similarity}')
