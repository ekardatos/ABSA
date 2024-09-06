import pandas as pd
import xml.etree.cElementTree as ET


def load_file(filename):
    # parsing the file
    tree = ET.parse(filename)
    root = tree.getroot()

    # setting the dataframe columns
    df_columns = ['Sentence', 'Opinion_target', 'Opinion_category',
                  'Opinion_polarity', 'Opinion_from', 'Opinion_to', 'Class']

    rows = []

    for node in root:
        sentences = node.find("sentences")
        for sentence in sentences:
            text_sentence = sentence.find('text').text
            opinions = sentence.find('Opinions')
            if opinions:
                for opinion in opinions:
                    opinion_target = opinion.get('target')
                    opinion_category = opinion.get('category')
                    opinion_polarity = opinion.get('polarity')
                    opinion_from = opinion.get('from')
                    opinion_to = opinion.get('to')

                    if opinion_polarity == 'positive':
                        sentence_class = 0
                    elif opinion_polarity == 'neutral':
                        sentence_class = 1
                    elif opinion_polarity == 'negative':
                        sentence_class = 2
                    else:
                        continue
                    rows.append({'Sentence': text_sentence, 'Opinion_target': opinion_target,
                                 'Opinion_category': opinion_category, 'Opinion_polarity': opinion_polarity,
                                 'Opinion_from': opinion_from, 'Opinion_to': opinion_to,
                                 'Class': sentence_class})

    df = pd.DataFrame(rows, columns=df_columns)
    return df

