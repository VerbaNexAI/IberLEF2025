from  text_processing import TextProcessing
from feature_extraction import FeatureExtraction

lang = 'es'
text = 'Vine a ver si se habían muerto en el apocalipsis y aún sigo leyendo sus tuits bien de la chingada'

ta = TextProcessing(lang=lang)
fe = FeatureExtraction(lang=lang)

text_clean = ta.transformer(text)

print('Original text: {0}'.format(text_clean))

print('Get Lexical Features')
print(fe.get_features_lexical(text_clean))