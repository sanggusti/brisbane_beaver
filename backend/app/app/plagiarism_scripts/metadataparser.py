'''
Author: Louis Owen (https://louisowen6.github.io/)
'''

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfparser import PDFDocument
from datetime import datetime
import docx


def getDocxMeta(path):
    doc = docx.Document(path)
    
    metadata = {}
    prop = doc.core_properties
    metadata["author"] = prop.author
    metadata["created"] = prop.created
    metadata["modified"] = prop.modified
    metadata["subject"] = prop.subject
    metadata["title"] = prop.title
    
    return metadata

def getPDFMeta(path):
    parser = PDFParser(open(path, 'rb'))
    doc = PDFDocument(parser)
    parser.set_document(doc)
    doc.set_parser(parser)
    
    metadata_dict = {}
    if len(doc.info) > 0:
        metadata_dict_temp = doc.info[0]

        metadata_dict['author'] = metadata_dict_temp['Author']
        metadata_dict['created'] = datetime.strptime(metadata_dict_temp['CreationDate'].split(':')[1].replace("'", ""), "%Y%m%d%H%M%S%z").replace(tzinfo=None)
        metadata_dict['modified'] = metadata_dict['created'] if metadata_dict_temp['ModDate']=='' else datetime.strptime(metadata_dict_temp['ModDate'].split(':')[1].replace("'", ""), "%Y%m%d%H%M%S%z").replace(tzinfo=None)
        metadata_dict['subject'] = metadata_dict_temp['Subject']
        metadata_dict['title'] = metadata_dict_temp['Title']
        
    return metadata_dict
    