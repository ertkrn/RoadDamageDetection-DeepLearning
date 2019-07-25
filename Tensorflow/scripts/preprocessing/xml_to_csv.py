import os
import glob
import pandas as pd
import argparse
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    print('Path: ', path)
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            if (len(member)>2):
                value = (root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text)
                    )
            else :
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[1][0].text),
                         int(member[1][1].text),
                         int(member[1][2].text),
                         int(member[1][3].text)
                         )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    path = '.../RoadDamageDetection-DeepLearning/Tensorflow/workspace/training_demo/images/'
    path_annotation = '.../RoadDamageDetection-DeepLearning/Tensorflow/workspace/training_demo/annotations/'
    for i in ['train', 'test']:
        image_path = os.path.join(path, '{}'.format(i))
        xml_df = xml_to_csv(image_path)
        csvpath = path_annotation + '{}_labels.csv'.format(i)
        print(csvpath)
        xml_df.to_csv(csvpath, index=None)
        print('Successfully converted xml to csv.')

if __name__ == '__main__':
    main()