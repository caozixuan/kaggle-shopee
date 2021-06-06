DATA_CONFIG = {
    'image_path': 'data/images',
    'train_images_filename': 'train_images',
    'test_images_filename': 'train_images',  # 'test_images',
    'match_path': 'data/match',
    'train_match_set_name': 'train.csv',
    'test_match_set_name': 'train.csv' # 'test.csv'
}

IMAGE_DATA_CONFIG = {
    'positive_sample_size': 1,
    'negative_sample_size': 5
}

TITLE_DATA_CONFIG = {
    'embed_size': 128,
    'positive_sample_size': 1,
    'negative_sample_size': 5,
    'embed_save_path': './pca_embed.pt'
}