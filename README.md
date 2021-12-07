# RU-BiLSTM-CRF
Implementation of Bidierctional LSTM + CRF (Conditional Random Field) for Named Entity Recognition in Russian.

Русская версия описания [тут](README_RU.md).

[![forthebadge](https://forthebadge.com/images/badges/designed-in-ms-paint.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/contains-tasty-spaghetti-code.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/powered-by-black-magic.svg)](https://forthebadge.com)

## Requirements:
1. [PyTorch](https://pytorch.org/docs/stable/index.html)
2. [AllenNLP](https://docs.allennlp.org/main/)*
3. [AllenNLP models](https://docs.allennlp.org/models/v1.1.0/)

***Note**: AllenNLP package already contains PyTorch but with no GPU support.  
Install any compatible with your GPU version of PyTorch from [here](https://pytorch.org/get-started/locally/).

## Dataset info:
[nerus](https://github.com/natasha/nerus/) from [natasha](https://github.com/natasha) project dataset was used for training.  
It contains over 700k texts from Lenta.ru.  
Markup is stored in the standard [CoNLL-U](https://universaldependencies.org/format.html) format.
```
$ zcat data/dataset/dataset.conllu.gz | head -40
# newdoc id = 0
# sent_id = 0_0
# text = Вице-премьер по социальным вопросам Татьяна Голикова рассказала, в каких регионах России зафиксирована наиболее высокая смертность от рака, сообщает РИА Новости.
1       Вице-премьер    _       NOUN    _       Animacy=Anim|Case=Nom|Gender=Masc|Number=Sing        7       nsubj   _       Tag=O
2       по      _       ADP     _       _       4       case    _       Tag=O
3       социальным      _       ADJ     _       Case=Dat|Degree=Pos|Number=Plur 4   amod     _       Tag=O
4       вопросам        _       NOUN    _       Animacy=Inan|Case=Dat|Gender=Masc|Number=Plur        1       nmod    _       Tag=O
5       Татьяна _       PROPN   _       Animacy=Anim|Case=Nom|Gender=Fem|Number=Sing1appos   _       Tag=B-PER
6       Голикова        _       PROPN   _       Animacy=Anim|Case=Nom|Gender=Fem|Number=Sing 5       flat:name       _       Tag=I-PER
7       рассказала      _       VERB    _       Aspect=Perf|Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act        0       root    _       Tag=O
8       ,       _       PUNCT   _       _       13      punct   _       Tag=O
9       в       _       ADP     _       _       11      case    _       Tag=O
10      каких   _       DET     _       Case=Loc|Number=Plur    11      det     _   Tag=O
11      регионах        _       NOUN    _       Animacy=Inan|Case=Loc|Gender=Masc|Number=Plur        13      obl     _       Tag=O
12      России  _       PROPN   _       Animacy=Inan|Case=Gen|Gender=Fem|Number=Sing11       nmod    _       Tag=B-LOC
13      зафиксирована   _       VERB    _       Aspect=Perf|Gender=Fem|Number=Sing|Tense=Past|Variant=Short|VerbForm=Part|Voice=Pass 7       ccomp   _       Tag=O
14      наиболее        _       ADV     _       Degree=Pos      15      advmod  _   Tag=O
15      высокая _       ADJ     _       Case=Nom|Degree=Pos|Gender=Fem|Number=Sing  16       amod    _       Tag=O
16      смертность      _       NOUN    _       Animacy=Inan|Case=Nom|Gender=Fem|Number=Sing 13      nsubj:pass      _       Tag=O
17      от      _       ADP     _       _       18      case    _       Tag=O
18      рака    _       NOUN    _       Animacy=Inan|Case=Gen|Gender=Masc|Number=Sing16      nmod    _       Tag=O
19      ,       _       PUNCT   _       _       20      punct   _       Tag=O
20      сообщает        _       VERB    _       Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act   0       root    _       Tag=O
21      РИА     _       PROPN   _       Animacy=Inan|Case=Nom|Gender=Neut|Number=Sing20      nsubj   _       Tag=B-ORG
22      Новости _       PROPN   _       Animacy=Inan|Case=Nom|Gender=Fem|Number=Plur21       appos   _       Tag=I-ORG
23      .       _       PUNCT   _       _       20      punct   _       Tag=O

# sent_id = 0_1
# text = По словам Голиковой, чаще всего онкологические заболевания становились причиной смерти в Псковской, Тверской, Тульской и Орловской областях, а также в Севастополе.
1       По      _       ADP     _       _       2       case    _       Tag=O
2       словам  _       NOUN    _       Animacy=Inan|Case=Dat|Gender=Neut|Number=Plur9       parataxis       _       Tag=O
3       Голиковой       _       PROPN   _       Animacy=Anim|Case=Gen|Gender=Fem|Number=Sing 2       nmod    _       Tag=B-PER
4       ,       _       PUNCT   _       _       2       punct   _       Tag=O
5       чаще    _       ADV     _       Degree=Cmp      9       advmod  _       Tag=O
6       всего   _       PRON    _       Animacy=Inan|Case=Gen|Gender=Neut|Number=Sing5       obl     _       Tag=O
7       онкологические  _       ADJ     _       Case=Nom|Degree=Pos|Number=Plur 8   amod     _       Tag=O
8       заболевания     _       NOUN    _       Animacy=Inan|Case=Nom|Gender=Neut|Number=Plur        9       nsubj   _       Tag=O
9       становились     _       VERB    _       Aspect=Imp|Mood=Ind|Number=Plur|Tense=Past|VerbForm=Fin|Voice=Mid    0       root    _       Tag=O
10      причиной        _       NOUN    _       Animacy=Inan|Case=Ins|Gender=Fem|Number=Sing 9       xcomp   _       Tag=O
11      смерти  _       NOUN    _       Animacy=Inan|Case=Gen|Gender=Fem|Number=Sing10       nmod    _       Tag=O
```

### Download:
[nerus_lenta.conllu.gz](https://storage.yandexcloud.net/natasha-nerus/data/nerus_lenta.conllu.gz)


## Embeddings:
Neural network can be trained with [ELMo](https://en.wikipedia.org/wiki/ELMo) embeddings.  
You can download any pretrained model from [here](http://vectors.nlpl.eu/repository/20/212.zip).

## Usage:
1. Download dataset and split it to train/test/validation subsets.
2. Download ELMo embeddings if you are going to use them.
3. Configure [config.json](config.json). Specify name of your model, paths to datasets/embeddings, model serialization and checkpoints paths, model and training parameters.
4. Run [main.py](main.py)
5. Wait until training is done.
6. Enjoy!
