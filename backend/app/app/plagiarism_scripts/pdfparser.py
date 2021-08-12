'''
References: https://github.com/Layout-Parser/layout-parser/blob/master/examples/Deep%20Layout%20Parsing.ipynb
'''

import pandas as pd
import numpy as np
import re

from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

import layoutparser as lp
from layoutparser.elements import Rectangle


def pdf_to_text(pdf_path):
    # Convert PDF to Images
    images = convert_from_path(pdf_path)
    images = [np.asarray(images[i]) for i in range(len(images))]
    
    # Load the deep layout model from the layoutparser API 
    # For all the supported model, please check the Model 
    # Zoo Page: https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html

    model = lp.Detectron2LayoutModel(
                                     'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', 
                                     extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                     label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

    # Initialize the tesseract ocr engine
    ocr_agent = lp.TesseractAgent(languages='ind')
    
    txt_list = []
    for image in images:
        text_blocks = extract_paragraphs(image,model,ocr_agent)

        if text_blocks is not None:
            for txt in text_blocks.get_texts():
                txt = re.sub('\n',' ',txt).strip()
                txt_list.append(re.sub('\x0c','',txt))
    
    #Append splitted paragraph
    excluded_id = []
    new_txt_list = []
    for i in range(len(txt_list)-1):
        if txt_list[i+1][0] != txt_list[i+1][0].lower():
            if i not in excluded_id:
                new_txt_list.append(txt_list[i])
        else:
            if i not in excluded_id:
                new_txt_list.append(txt_list[i]+' '+txt_list[i+1])

                excluded_id.append(i+1)

    if (len(txt_list) - 1) not in excluded_id:
        new_txt_list.append(txt_list[-1])
    
    return new_txt_list
    
    
def extract_paragraphs(image,model,ocr_agent):
    # Detect the layout of the input image
    layout = model.detect(image)
    
    #Firstly we filter text region of specific type:
    text_blocks = lp.Layout([b for b in layout if b.type=='Text'])
    figure_blocks = lp.Layout([b for b in layout if b.type=='Title'])
    
    #As there could be text region detected inside the figure region, we just drop them:
    text_blocks = lp.Layout([b for b in text_blocks \
                       if not any(b.is_in(b_fig) for b_fig in figure_blocks)])
    
    if len(text_blocks)>0:
        # Indexing
        y1_loc_list = []
        for block in text_blocks:
            y1_loc_list.append(block.to_dict()['y_1'])

        id_list = (pd.Series(y1_loc_list).rank()-1).astype(int).to_list()

        text_blocks = lp.Layout([b.set(id = id_list[idx]) for idx, b in enumerate(text_blocks)])


        # Sorting
        text_blocks = lp.Layout([x for _,x in sorted(zip(id_list,text_blocks))])

        # Expanding
        y1_loc_list = []
        y2_loc_list = []
        for block in text_blocks:
            y1_loc_list.append(block.to_dict()['y_1'])
            y2_loc_list.append(block.to_dict()['y_2'])

        excluded_id = []
        expanded_text_blocks = []
        for i in range(len(y2_loc_list)-1):
            if (y1_loc_list[i]>=y1_loc_list[i+1]) and (y1_loc_list[i]<=y2_loc_list[i+1]):
                block = text_blocks[i]
                next_block = text_blocks[i+1]

                block.set(block=Rectangle(x_1=min(block.to_dict()['x_1'],next_block.to_dict()['x_1']),
                                          x_2=max(block.to_dict()['x_2'],next_block.to_dict()['x_2']),
                                          y_1=min(block.to_dict()['y_1'],next_block.to_dict()['y_1']),
                                          y_2=max(block.to_dict()['y_2'],next_block.to_dict()['y_2']))
                         , inplace=True)

                expanded_text_blocks.append(block)
                excluded_id.append(i+1)
            elif (y1_loc_list[i+1]>=y1_loc_list[i]) and (y1_loc_list[i+1]<=y2_loc_list[i]):
                block = text_blocks[i]
                next_block = text_blocks[i+1]

                block.set(block=Rectangle(x_1=min(block.to_dict()['x_1'],next_block.to_dict()['x_1']),
                                          x_2=max(block.to_dict()['x_2'],next_block.to_dict()['x_2']),
                                          y_1=min(block.to_dict()['y_1'],next_block.to_dict()['y_1']),
                                          y_2=max(block.to_dict()['y_2'],next_block.to_dict()['y_2']))
                         , inplace=True)

                expanded_text_blocks.append(block)
                excluded_id.append(i+1)
            else:
                if i not in excluded_id:
                    expanded_text_blocks.append(text_blocks[i])

        if (len(y2_loc_list) - 1) not in excluded_id:
            expanded_text_blocks.append(text_blocks[len(text_blocks)-1])

        expanded_text_blocks = lp.Layout(expanded_text_blocks)

        for block in expanded_text_blocks:
            # add padding in each image segment can help improve robustness 
            segment_image = (block
                               .pad(left=5, right=5, top=5, bottom=5)
                               .crop_image(image))


            text = ocr_agent.detect(segment_image)
            block.set(text=text, inplace=True)

        return expanded_text_blocks
    else:
        return None